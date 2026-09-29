//! Time `ProtoArray::is_payload_invalid` on the linear chains that fork choice
//! already uses for long non-finality.
//!
//! Chain shape matches `benches/find_head.rs`. Sizes come from `PAYLOAD_INVALID_SIZES`
//! (comma-separated). Default is `10000` so a bare run does not build a month-long chain.
//!
//! ```text
//! PAYLOAD_INVALID_SIZES=10000,50000 \
//!   cargo bench -p proto_array --bench is_payload_invalid
//! ```
//!
//! Each call walks every node. Pre-Gloas `build_chain` stores `Optimistic(zero)` on every
//! node, so querying zero matches the whole vec. This bench retargets one node to a unique
//! hash and queries that. A miss is that hash while it is not `Invalid`. A hit is the same
//! hash after that one node is marked `Invalid`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fixed_bytes::FixedBytesExtended;
use proto_array::{Block, ExecutionStatus, PayloadBlockHash, ProtoArrayForkChoice};
use std::hint::black_box;
use std::time::Duration;
use types::{
    AttestationShufflingId, Checkpoint, Epoch, EthSpec, ExecutionBlockHash, Hash256,
    MainnetEthSpec, Slot,
};

fn get_root(i: u64) -> Hash256 {
    Hash256::from_low_u64_be(i)
}

fn get_hash(i: u64) -> ExecutionBlockHash {
    ExecutionBlockHash::from_root(get_root(i))
}

/// Build a linear chain of `num_blocks` blocks.
///
/// Kept aligned with `benches/find_head.rs`.
fn build_chain(num_blocks: u64, gloas: bool) -> (ProtoArrayForkChoice, types::ChainSpec) {
    let mut spec = MainnetEthSpec::default_spec();
    let gloas_fork_slot = 32;
    if gloas {
        spec.gloas_fork_epoch = Some(Epoch::new(1));
    }

    let finalized_checkpoint = Checkpoint {
        epoch: Epoch::new(0),
        root: get_root(0),
    };
    let junk_shuffling_id = AttestationShufflingId::from_components(Epoch::new(0), Hash256::zero());

    let mut fork_choice = ProtoArrayForkChoice::new::<MainnetEthSpec>(
        Slot::new(0),
        Slot::new(0),
        Hash256::zero(),
        finalized_checkpoint,
        finalized_checkpoint,
        junk_shuffling_id.clone(),
        junk_shuffling_id.clone(),
        ExecutionStatus::Optimistic(ExecutionBlockHash::zero()),
        None,
        None,
        0,
        &spec,
    )
    .expect("should create fork choice");

    for i in 1..=num_blocks {
        let is_gloas = gloas && i >= gloas_fork_slot;
        let block = Block {
            slot: Slot::new(i),
            root: get_root(i),
            parent_root: Some(get_root(i - 1)),
            state_root: Hash256::zero(),
            target_root: get_root(0),
            current_epoch_shuffling_id: junk_shuffling_id.clone(),
            next_epoch_shuffling_id: junk_shuffling_id.clone(),
            justified_checkpoint: finalized_checkpoint,
            finalized_checkpoint,
            execution_status: ExecutionStatus::Optimistic(ExecutionBlockHash::zero()),
            unrealized_justified_checkpoint: Some(finalized_checkpoint),
            unrealized_finalized_checkpoint: Some(finalized_checkpoint),
            execution_payload_parent_hash: if is_gloas {
                Some(get_hash(i - 1))
            } else {
                None
            },
            execution_payload_block_hash: if is_gloas { Some(get_hash(i)) } else { None },
            proposer_index: Some(0),
            payload_received: false,
        };

        fork_choice
            .process_block::<MainnetEthSpec>(block, Slot::new(i), &spec, Duration::ZERO)
            .expect("should process block");
    }

    (fork_choice, spec)
}

fn sizes_from_env() -> Vec<u64> {
    let raw = std::env::var("PAYLOAD_INVALID_SIZES").unwrap_or_else(|_| "10000".to_string());
    raw.split(',')
        .map(|part| {
            part.trim()
                .parse::<u64>()
                .expect("PAYLOAD_INVALID_SIZES entries must be integers")
        })
        .inspect(|size| {
            assert!(
                *size >= 64,
                "size {size} is below the Gloas fork slot used by this chain"
            );
        })
        .collect()
}

fn block_hash_at(fork_choice: &ProtoArrayForkChoice, index: usize) -> ExecutionBlockHash {
    match fork_choice.core_proto_array().nodes[index].block_hash() {
        PayloadBlockHash::Hash(hash) => hash,
        PayloadBlockHash::PreMerge => panic!("node {index} has no execution hash"),
    }
}

fn set_status(fork_choice: &mut ProtoArrayForkChoice, index: usize, status: ExecutionStatus) {
    let node = fork_choice
        .core_proto_array_mut()
        .nodes
        .get_mut(index)
        .expect("node index");
    *node.execution_status_mut() = status;
}

fn is_payload_invalid(fork_choice: &ProtoArrayForkChoice, hash: ExecutionBlockHash) -> bool {
    fork_choice.core_proto_array().is_payload_invalid(&hash)
}

/// Point the timed lookup at one payload hash that exists on exactly one node.
fn prepare_query(
    fork_choice: &mut ProtoArrayForkChoice,
    num_blocks: u64,
    gloas: bool,
) -> (usize, ExecutionBlockHash) {
    let index = (num_blocks / 2) as usize;
    let nodes = &fork_choice.core_proto_array().nodes;
    assert!(index < nodes.len(), "chain is shorter than {num_blocks}");

    let hash = if gloas {
        let hash = block_hash_at(fork_choice, index);
        assert_eq!(
            hash,
            get_hash(index as u64),
            "Gloas node {index} should commit to get_hash({index})"
        );
        hash
    } else {
        let zeros = fork_choice
            .core_proto_array()
            .execution_block_hash_to_node_indices(&ExecutionBlockHash::zero());
        assert_eq!(
            zeros.len(),
            fork_choice.core_proto_array().nodes.len(),
            "pre-Gloas build stores the zero hash on every node"
        );
        // Distinct from zero, and not equal to a Gloas-style per-slot hash already stored.
        let hash = get_hash(num_blocks.saturating_add(1));
        set_status(fork_choice, index, ExecutionStatus::Optimistic(hash));
        let zeros = fork_choice
            .core_proto_array()
            .execution_block_hash_to_node_indices(&ExecutionBlockHash::zero());
        assert_eq!(
            zeros.len() + 1,
            fork_choice.core_proto_array().nodes.len(),
            "exactly one pre-Gloas node was retargeted"
        );
        hash
    };

    let matches = fork_choice
        .core_proto_array()
        .execution_block_hash_to_node_indices(&hash);
    assert_eq!(matches.len(), 1, "query hash must match exactly one node");
    assert!(
        !is_payload_invalid(fork_choice, hash),
        "miss setup must not already be invalid"
    );
    (index, hash)
}

fn bench_is_payload_invalid(c: &mut Criterion) {
    let node_size = std::mem::size_of::<proto_array::core::ProtoNode>();
    eprintln!("ProtoNode size_of = {node_size} bytes");

    let mut group = c.benchmark_group("is_payload_invalid");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for (label, gloas) in [("pre_gloas", false), ("gloas", true)] {
        for num_blocks in sizes_from_env() {
            eprintln!("building {label} chain of {num_blocks} blocks");
            let (mut fork_choice, _spec) = build_chain(num_blocks, gloas);
            let node_count = fork_choice.core_proto_array().nodes.len() as u64;
            eprintln!(
                "{label} nodes = {node_count}, vec bytes ~= {}",
                node_count.saturating_mul(node_size as u64)
            );

            let (index, hash) = prepare_query(&mut fork_choice, num_blocks, gloas);

            group.throughput(Throughput::Elements(node_count));
            group.bench_function(BenchmarkId::new(format!("{label}_miss"), num_blocks), |b| {
                b.iter(|| black_box(is_payload_invalid(black_box(&fork_choice), black_box(hash))));
            });

            set_status(&mut fork_choice, index, ExecutionStatus::Invalid(hash));
            assert!(
                is_payload_invalid(&fork_choice, hash),
                "hit setup must observe the invalid hash"
            );
            let matches = fork_choice
                .core_proto_array()
                .execution_block_hash_to_node_indices(&hash);
            assert_eq!(matches.len(), 1, "invalidation retargeted a single node");

            group.bench_function(BenchmarkId::new(format!("{label}_hit"), num_blocks), |b| {
                b.iter(|| black_box(is_payload_invalid(black_box(&fork_choice), black_box(hash))));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_is_payload_invalid);
criterion_main!(benches);
