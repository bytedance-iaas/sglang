#pragma once

#include "sm_copy.cuh"

#include <memory>
#include <vector>

#if CUDART_VERSION < 12080
#error "SiDP conditional DMA requires CUDA Toolkit 12.8 or newer (SWITCH nodes)"
#endif

namespace sidp {

inline void dma_graph_check(cudaError_t error) {
  host::RuntimeCheck(error == cudaSuccess, "SiDP conditional DMA: ", cudaGetErrorString(error));
}

// Fail-stop, never silently skip a copy by supplying an out-of-range SWITCH value.
__global__ void set_dma_condition_kernel(
    cudaGraphConditionalHandle handle,
    const int32_t* selected,
    int32_t count,
    int32_t* error_state) {
  const int32_t index = selected[0];
  if (index < 0 || index >= count) fail_stop(error_state, kErrorInvalidSelection);
  cudaGraphSetConditional(handle, static_cast<unsigned>(index));
}

__global__ void set_dma_condition_traced_kernel(
    cudaGraphConditionalHandle handle,
    const int32_t* selected,
    int32_t count,
    int32_t* error_state,
    SmExecutionTrace trace) {
  const int32_t trace_index = begin_sm_trace(trace);
  const int32_t index = selected[0];
  if (index < 0 || index >= count) fail_stop(error_state, kErrorInvalidSelection);
  cudaGraphSetConditional(handle, static_cast<unsigned>(index));
  end_sm_trace(trace, trace_index);
}

// Each call adds nodes directly to the parent/body graph. Conditional graphs
// must not be cloned as reusable child graphs. CUDA copies kernel arguments
// during construction, so these temporary host argument addresses are safe.
template <typename Kernel, typename... Args>
cudaGraphNode_t add_dma_kernel(
    cudaGraph_t graph, cudaGraphNode_t dependency, unsigned threads, Kernel kernel, Args... args) {
  void* arguments[] = {static_cast<void*>(&args)...};
  cudaKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(kernel);
  params.gridDim = dim3(1);
  params.blockDim = dim3(threads);
  params.kernelParams = arguments;
  cudaGraphNode_t node;
  dma_graph_check(cudaGraphAddKernelNode(
      &node, graph, dependency ? &dependency : nullptr, dependency ? 1 : 0, &params));
  return node;
}

struct DmaGraphHandle {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  int device = -1;

  ~DmaGraphHandle() {
    // Destruction follows stream quiescence in the Python owner. Also clean up
    // partial construction on exceptions without masking the original error.
    int previous = -1;
    cudaGetDevice(&previous);
    if (device >= 0) cudaSetDevice(device);
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
  }
};

struct SidpDmaGraphKernels {
  // A single node builder for standalone eager graphs and in-place model
  // capture. Each invocation creates handles in its own parent graph: graphs
  // containing conditional nodes cannot be cloned as reusable child graphs.
  template <bool Capturing>
  static int64_t build(
      tvm::ffi::TensorView owner_state_ptrs,
      tvm::ffi::TensorView candidate_owners,
      tvm::ffi::TensorView candidate_slots,
      tvm::ffi::TensorView done,
      tvm::ffi::TensorView fill_gen,
      tvm::ffi::TensorView comp_gen,
      tvm::ffi::TensorView probe_cursor,
      tvm::ffi::TensorView selected,
      tvm::ffi::TensorView claim_spins,
      tvm::ffi::TensorView claim_collisions,
      tvm::ffi::TensorView error_state,
      tvm::ffi::TensorView copies,
      int64_t required_comp_gen,
      int64_t target_fill_gen,
      int64_t claim_order,
      int64_t requester_rank,
      int64_t backoff_ns,
      int64_t timeout_clocks,
      tvm::ffi::TensorView selected_trace,
      tvm::ffi::TensorView spins_trace,
      tvm::ffi::TensorView collisions_trace,
      tvm::ffi::TensorView timing_events,
      int64_t trace_offset,
      tvm::ffi::TensorView sm_trace_rows,
      tvm::ffi::TensorView sm_trace_count,
      tvm::ffi::TensorView sm_trace_overflow,
      int64_t sm_trace_tag_base) {
    using namespace host;
    auto D = SymbolicSize{"owners"};
    auto C = SymbolicSize{"candidates"};
    auto S = SymbolicSize{"slots"};
    auto P = SymbolicSize{"components"};
    auto device = SymbolicDevice{};
    TensorMatcher({D}).with_dtype<uint64_t>().with_device<kDLGPU>(device).verify(owner_state_ptrs);
    TensorMatcher({C}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(candidate_owners).verify(candidate_slots);
    TensorMatcher({C}).with_dtype<uint8_t>().with_device<kDLGPU>(device).verify(done);
    TensorMatcher({S}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(fill_gen).verify(comp_gen);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(probe_cursor).verify(selected).verify(error_state);
    TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(claim_spins).verify(claim_collisions);
    // Fixed addresses, assembled on the host at setup; never copied back from
    // the GPU and never dereferenced by the host during a forward.
    TensorMatcher({P, C, 3}).with_dtype<uint64_t>().with_device<kDLCPU>().verify(copies);
    RuntimeCheck(C.unwrap() > 0 && C.unwrap() <= 256 && P.unwrap() > 0, "invalid DMA cycle shape");
    RuntimeCheck(requester_rank >= 0 && requester_rank < D.unwrap(), "invalid requester rank");
    RuntimeCheck(required_comp_gen >= 0 && target_fill_gen > required_comp_gen, "invalid DMA generations");
    RuntimeCheck(
        claim_order == kClaimOrderRotating || claim_order == kClaimOrderComputePriority,
        "invalid SiDP dynamic claim order");
    const bool profiling = trace_offset >= 0;
    if (profiling) {
      auto T = SymbolicSize{"trace_size"};
      TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected_trace);
      TensorMatcher({T}).with_dtype<int64_t>().with_device<kDLGPU>(device).verify(spins_trace).verify(collisions_trace);
      TensorMatcher({C, 3}).with_dtype<uint64_t>().with_device<kDLCPU>().verify(timing_events);
      RuntimeCheck(trace_offset + C.unwrap() <= T.unwrap(), "invalid DMA trace range");
    }
    const bool sm_profiling = sm_trace_tag_base >= 0;
    SmExecutionTrace sm_trace{};
    if (sm_profiling) {
      sm_trace = SidpSmCopyKernels::validate_sm_trace(
          sm_trace_rows,
          sm_trace_count,
          sm_trace_overflow,
          sm_trace_tag_base,
          device);
    }
    auto tagged_trace = [&](int64_t tag) {
      auto result = sm_trace;
      result.tag = static_cast<int32_t>(tag);
      return result;
    };

    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    cudaStreamCaptureStatus capture_status;
    dma_graph_check(cudaStreamIsCapturing(stream, &capture_status));
    cudaGraph_t graph = nullptr;
    std::unique_ptr<DmaGraphHandle> result;
    const cudaGraphNode_t* dependencies = nullptr;
    const cudaGraphEdgeData* edge_data = nullptr;
    size_t num_dependencies = 0;
    if constexpr (Capturing) {
      RuntimeCheck(capture_status == cudaStreamCaptureStatusActive,
                   "SiDP append_cycle_to_capture requires an actively capturing comm stream");
      unsigned long long capture_id = 0;
#if CUDART_VERSION >= 13000
      dma_graph_check(cudaStreamGetCaptureInfo(
#else
      dma_graph_check(cudaStreamGetCaptureInfo_v3(
#endif
          stream, &capture_status, &capture_id, &graph, &dependencies, &edge_data, &num_dependencies));
      // Borrowed graph/dependency arrays: only graph APIs are used until the
      // incoming edges are attached below. Never destroy this model graph.
    } else {
      RuntimeCheck(capture_status == cudaStreamCaptureStatusNone, "DMA graphs must be built during setup");
      result = std::make_unique<DmaGraphHandle>();
      dma_graph_check(cudaGetDevice(&result->device));
      RuntimeCheck(result->device == device.unwrap().device_id, "DMA graph must be built on requester device");
      int driver_version = 0;
      dma_graph_check(cudaDriverGetVersion(&driver_version));
      RuntimeCheck(driver_version >= 12080, "SiDP conditional DMA requires a CUDA 12.8+ driver");
      dma_graph_check(cudaGraphCreate(&result->graph, 0));
      graph = result->graph;
    }

    const auto owner_ptrs = static_cast<const uint64_t*>(owner_state_ptrs.data_ptr());
    const auto owners = static_cast<const int32_t*>(candidate_owners.data_ptr());
    const auto slots = static_cast<const int32_t*>(candidate_slots.data_ptr());
    auto done_ptr = static_cast<uint8_t*>(done.data_ptr());
    auto fill_ptr = static_cast<int32_t*>(fill_gen.data_ptr());
    auto comp_ptr = static_cast<int32_t*>(comp_gen.data_ptr());
    auto cursor = static_cast<int32_t*>(probe_cursor.data_ptr());
    auto selection = static_cast<int32_t*>(selected.data_ptr());
    auto spins = static_cast<int64_t*>(claim_spins.data_ptr());
    auto collisions = static_cast<int64_t*>(claim_collisions.data_ptr());
    auto error = static_cast<int32_t*>(error_state.data_ptr());
    const int32_t count = static_cast<int32_t>(C.unwrap());
    const int32_t owner_count = static_cast<int32_t>(D.unwrap());
    const int32_t requester = static_cast<int32_t>(requester_rank);
    const auto descriptors = static_cast<const uint64_t*>(copies.data_ptr());

    cudaGraphNode_t tail;
    if (sm_profiling) {
      tail = add_dma_kernel(
          graph,
          nullptr,
          256,
          reset_cycle_state_traced_kernel,
          done_ptr,
          count,
          selection,
          spins,
          collisions,
          tagged_trace(sm_trace_tag_base));
    } else {
      tail = add_dma_kernel(graph, nullptr, 256, reset_cycle_state_kernel,
                            done_ptr, count, selection, spins, collisions);
    }
    if (num_dependencies) {
      // Preserve every incoming edge, including non-default edge metadata.
      // The single-dependency helper remains sufficient for the chain itself.
      std::vector<cudaGraphNode_t> targets(num_dependencies, tail);
#if CUDART_VERSION >= 13000
      dma_graph_check(cudaGraphAddDependencies(
#else
      dma_graph_check(cudaGraphAddDependencies_v2(
#endif
          graph, dependencies, targets.data(), edge_data, num_dependencies));
    }
    const auto events = profiling ? static_cast<const uint64_t*>(timing_events.data_ptr()) : nullptr;
    auto record_event = [&](cudaGraphNode_t dependency, int32_t step, int32_t marker) {
      if (!profiling) return dependency;
      auto event = reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(events[step * 3 + marker]));
      RuntimeCheck(event != nullptr, "uninitialized DMA profiling event");
      cudaGraphNode_t node;
      // EventRecord is legal only in the parent, never in a conditional body.
      dma_graph_check(cudaGraphAddEventRecordNode(&node, graph, &dependency, 1, event));
      return node;
    };
    // Unroll the known number of steps. Only the selected candidate is dynamic.
    // Every step releases its owner and publishes its slot before the next
    // claim; no whole-cycle owner lock or whole-cycle RAW/WAR barrier is added.
    for (int32_t step = 0; step < count; ++step) {
      tail = record_event(tail, step, 0);
      const int64_t step_tag = sm_trace_tag_base + 1 + step * 4;
      if (sm_profiling) {
        tail = add_dma_kernel(graph, tail, 1, claim_owner_traced_kernel,
            owner_ptrs, owner_count, owners, slots, done_ptr, count, comp_ptr,
            static_cast<int32_t>(required_comp_gen), static_cast<int32_t>(claim_order),
            cursor, selection, spins, collisions,
            requester, static_cast<int32_t>(backoff_ns), static_cast<uint64_t>(timeout_clocks), error,
            tagged_trace(step_tag));
      } else {
        tail = add_dma_kernel(graph, tail, 1, claim_owner_kernel,
            owner_ptrs, owner_count, owners, slots, done_ptr, count, comp_ptr,
            static_cast<int32_t>(required_comp_gen), static_cast<int32_t>(claim_order),
            cursor, selection, spins, collisions,
            requester, static_cast<int32_t>(backoff_ns), static_cast<uint64_t>(timeout_clocks), error);
      }

      tail = record_event(tail, step, 1);
      if (profiling) {
        tail = add_dma_kernel(graph, tail, 1, record_trace_kernel,
            selection, spins, collisions, static_cast<int32_t*>(selected_trace.data_ptr()),
            static_cast<int64_t*>(spins_trace.data_ptr()), static_cast<int64_t*>(collisions_trace.data_ptr()),
            static_cast<int32_t>(trace_offset + step));
      }
      cudaGraphConditionalHandle condition;
      dma_graph_check(cudaGraphConditionalHandleCreate(&condition, graph, count, cudaGraphCondAssignDefault));
      if (sm_profiling) {
        tail = add_dma_kernel(
            graph,
            tail,
            1,
            set_dma_condition_traced_kernel,
            condition,
            selection,
            count,
            error,
            tagged_trace(step_tag + 1));
      } else {
        tail = add_dma_kernel(graph, tail, 1, set_dma_condition_kernel, condition, selection, count, error);
      }
      cudaGraphNodeParams params{};
      params.type = cudaGraphNodeTypeConditional;
      params.conditional.handle = condition;
      params.conditional.type = cudaGraphCondTypeSwitch;
      params.conditional.size = count;
      cudaGraphNode_t branch;
#if CUDART_VERSION >= 13000
      dma_graph_check(cudaGraphAddNode(&branch, graph, &tail, nullptr, 1, &params));
#else
      dma_graph_check(cudaGraphAddNode(&branch, graph, &tail, 1, &params));
#endif

      for (int32_t candidate = 0; candidate < count; ++candidate) {
        cudaGraph_t body = params.conditional.phGraph_out[candidate];
        cudaGraphNode_t copy_tail = nullptr;
        for (int64_t component = 0; component < P.unwrap(); ++component) {
          const auto desc = descriptors + (component * count + candidate) * 3;
          auto src = reinterpret_cast<const void*>(static_cast<uintptr_t>(desc[0]));
          auto dst = reinterpret_cast<void*>(static_cast<uintptr_t>(desc[1]));
          const size_t nbytes = static_cast<size_t>(desc[2]);
          if (nbytes == 0) continue;
          RuntimeCheck(src && dst, "null DMA component address");
          cudaGraphNode_t copy;
          dma_graph_check(cudaGraphAddMemcpyNode1D(
              &copy, body, copy_tail ? &copy_tail : nullptr, copy_tail ? 1 : 0,
              dst, src, nbytes, cudaMemcpyDefault));
          copy_tail = copy;
        }
        if (sm_profiling) {
          copy_tail = add_dma_kernel(body, copy_tail, 1, release_owner_traced_kernel,
              owner_ptrs, owner_count, owners, count, selection, requester, error,
              tagged_trace(step_tag + 2));
          add_dma_kernel(body, copy_tail, 1, publish_selected_fill_traced_kernel,
              fill_ptr, slots, count, selection, static_cast<int32_t>(target_fill_gen), error,
              tagged_trace(step_tag + 3));
        } else {
          copy_tail = add_dma_kernel(body, copy_tail, 1, release_owner_kernel,
              owner_ptrs, owner_count, owners, count, selection, requester, error);
          add_dma_kernel(body, copy_tail, 1, publish_selected_fill_kernel,
              fill_ptr, slots, count, selection, static_cast<int32_t>(target_fill_gen), error);
        }
      }
      tail = record_event(branch, step, 2);
    }
    if constexpr (Capturing) {
      // Subsequent captured comm work and the final compute-stream join must
      // depend on the whole SWITCH, including its release/publish body tail.
#if CUDART_VERSION >= 13000
      dma_graph_check(cudaStreamUpdateCaptureDependencies(
#else
      dma_graph_check(cudaStreamUpdateCaptureDependencies_v2(
#endif
          stream, &tail, nullptr, 1, cudaStreamSetCaptureDependencies));
      return 0;
    } else {
      dma_graph_check(cudaGraphInstantiate(&result->exec, graph, 0));
      // Keep source graph/conditional handles alive for the executable lifetime.
      dma_graph_check(cudaGraphUpload(result->exec, stream));
      return reinterpret_cast<int64_t>(result.release());
    }
  }

  static void launch(int64_t handle, tvm::ffi::TensorView selected) {
    using namespace host;
    auto device = SymbolicDevice{};
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(selected);
    auto* graph = reinterpret_cast<DmaGraphHandle*>(handle);
    RuntimeCheck(graph && graph->device == device.unwrap().device_id, "invalid DMA graph device");
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    cudaStreamCaptureStatus status;
    dma_graph_check(cudaStreamIsCapturing(stream, &status));
    RuntimeCheck(status == cudaStreamCaptureStatusNone,
                 "Cannot launch a standalone conditional DMA graph during capture; use append_cycle_to_capture");
    dma_graph_check(cudaGraphLaunch(graph->exec, stream));
  }

  static void destroy(int64_t handle) {
    delete reinterpret_cast<DmaGraphHandle*>(handle);
  }
};

}  // namespace sidp
