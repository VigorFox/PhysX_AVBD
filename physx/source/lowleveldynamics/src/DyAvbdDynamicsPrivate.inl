// AVBD context storage. Keep runtime knobs in AvbdSolverConfig
// (DyAvbdTypes.h); profiling state must remain task-local/cold.
  AvbdSolver mSolver; //!< AVBD solver instance
  PxcScratchAllocator &mScratchAllocator;   //!< Scratch memory allocator
  AvbdSoftIslandProvider
      *mSoftIslandProvider; //!< Scene-owned complete soft tuple provider
  AvbdRigidGpuWaveBackend
      *mRigidGpuWaveBackend; //!< Optional transactional owner-wave backend
  PxU32 mAvbdIterations; //!< Scene-wide complete AVBD iteration budget
  PxU32 mAvbdJointIterationOverride; //!< Optional joint-island minimum budget
  bool mAvbdEnableEarlyStop; //!< Allow pose-delta convergence termination

  class ScratchAllocatorAdapter : public PxAllocatorCallback {
  public:
    ScratchAllocatorAdapter(PxcScratchAllocator &scratch);
    virtual void *allocate(size_t size, const char *, const char *,
                           int) override;
    virtual void deallocate(void *) override;
    PxcScratchAllocator &mScratch;
  };
  ScratchAllocatorAdapter mScratchAdapter;

  class VirtualAllocatorAdapter : public PxAllocatorCallback {
  public:
    explicit VirtualAllocatorAdapter(Cm::VirtualAllocatorCallback &allocator);
    virtual void *allocate(size_t size, const char *, const char *file,
                           int line) override;
    virtual void deallocate(void *ptr) override;

  private:
    Cm::VirtualAllocatorCallback &mAllocator;
  };
  VirtualAllocatorAdapter mAllocatorAdapter;

  PxTaskManager *mTaskManager;   //!< Task manager for parallel execution
  AvbdTaskFactory *mTaskFactory; //!< Factory for creating AVBD tasks
  AvbdKernelLabCapture *mKernelLabCapture; //!< Opt-in lab-only capture sink
  // Serial update owns this latch. Once a task has been given the sole sink
  // ticket, later island submissions must not read the sink state while that
  // task's worker may be publishing its capture.
  bool mKernelLabCaptureReservationSubmitted;
  // P2 only supports the Scene PxTaskManager path. `true` is the explicit
  // serial reference mode selected by PHYSX_AVBD_TASKGRAPH_SERIAL.
  bool mTaskGraphSerialMode;
  // Task-graph and iteration diagnostics are intentionally kept out of the
  // solver context. Profile data is local to tasks and reduced after join.
  bool mFrictionEveryIteration; //!< Apply friction every iteration
  bool mSolverInitialized;      //!< Whether solver has been initialized

  //!< Track heap fallback allocations for cleanup at frame end
  //!< No mutex needed since update() and mergeResults() are called from
  //!< single-threaded contexts
  PxArray<void *> mHeapFallbackAllocations;
