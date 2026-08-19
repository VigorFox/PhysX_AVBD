// AVBD diagnostics and task-graph telemetry declarations.
// Included inside AvbdDynamicsContext's public section.
  // Runtime task profiling is intentionally out of the solver context.
  // These compatibility hooks are no-ops until task-local profile records
  // and join reduction are introduced. They must never enter the solver hot
  // path or update shared state.
  void beginIterationDiagnosticsFrame() {}
  void recordIterationDiagnostics(PxU32, const AvbdSolverStats &, bool,
                                  const AvbdD6JointConstraint * = nullptr,
                                  PxU32 = 0) {}
  void flushIterationDiagnosticsFrame() {}
  void resetTaskGraphStatistics() {}
  void recordSolveTaskSubmitted(const AvbdIslandBatch &) {}
  void recordStandaloneSoftSolveTaskSubmitted(
      PxU32, bool = false) {}
  void beginSolveTask() {}
  void endSolveTask() {}
  void recordSerialSolve(const AvbdIslandBatch &) {}
  void recordStandaloneSoftSerialSolve(PxU32) {}
  void recordPredictionTasksSubmitted(PxU32) {}
  void beginPredictionTask() {}
  void endPredictionTask() {}
  void recordSerialPredictionStage() {}
  void recordWriteBackTasksSubmitted(PxU32) {}
  void beginWriteBackTask() {}
  void endWriteBackTask() {}
  void recordSerialWriteBackStage() {}
  void recordCausalLayerTasksSubmitted(PxU32, PxU32) {}
  void beginCausalLayerTask() {}
  void endCausalLayerTask() {}
  void recordCausalLayerFanIn(PxReal) {}
  void recordSerialCausalLayerFallback(PxU32) {}
  void recordCausalLayerTaskPoolGrowth(PxU64) {}
  void recordWorldPlaneContactTasksSubmitted(PxU32) {}
  void beginWorldPlaneContactTask() {}
  void endWorldPlaneContactTask() {}
  void recordWorldPlaneContactFanIn() {}
  void recordSerialWorldPlaneContactFallback() {}
  void recordRigidBoxSdfContactTasksSubmitted(PxU32) {}
  void beginRigidBoxSdfContactTask() {}
  void endRigidBoxSdfContactTask() {}
  void recordRigidBoxSdfContactFanIn() {}
  void recordSerialRigidBoxSdfContactFallback() {}
  void recordRigidSphereSdfContactTasksSubmitted(PxU32) {}
  void beginRigidSphereSdfContactTask() {}
  void endRigidSphereSdfContactTask() {}
  void recordRigidSphereSdfContactFanIn() {}
  void recordSerialRigidSphereSdfContactFallback() {}
  void recordRigidCapsuleSdfContactTasksSubmitted(PxU32) {}
  void beginRigidCapsuleSdfContactTask() {}
  void endRigidCapsuleSdfContactTask() {}
  void recordRigidCapsuleSdfContactFanIn() {}
  void recordSerialRigidCapsuleSdfContactFallback() {}
  void recordRigidConvexSdfContactTasksSubmitted(PxU32) {}
  void beginRigidConvexSdfContactTask() {}
  void endRigidConvexSdfContactTask() {}
  void recordRigidConvexSdfContactFanIn() {}
  void recordSerialRigidConvexSdfContactFallback() {}
  void recordRigidTriangleSurfaceContactTasksSubmitted(PxU32) {}
  void beginRigidTriangleSurfaceContactTask() {}
  void endRigidTriangleSurfaceContactTask() {}
  void recordRigidTriangleSurfaceContactFanIn() {}
  void recordSerialRigidTriangleSurfaceContactFallback() {}
  void recordRigidTriangleSurfaceContactWork(
      PxU64, PxU64, PxU32, PxU64, PxU64, PxU32, PxU32, PxU64, PxU64,
      PxU64, PxU64, PxU64) {}
  void recordRigidTriangleSurfaceContactTaskPoolGrowth(PxU64) {}
  void recordRigidTriangleSurfaceContactOutputGrowth(PxU64) {}
  void recordRigidTriangleSurfaceContactQueryScratchGrowth(PxU64) {}
  void recordRigidTriangleSurfaceContactResidentCapacity(PxU64, PxU64,
                                                          PxU64) {}
  void recordRigidTriangleSurfaceContactTaskWallTime(PxU64) {}
  void recordRigidTriangleSurfaceContactFanInWallSpan(PxU64) {}
  void recordRigidTriangleSurfaceContactSerialTransactionWallTime(PxU64) {}
  void recordRigidTriangleSurfaceContactParentCompletionWallTime(PxU64) {}
  void recordRigidTriangleSurfaceContactPostContinuationWallTime(PxU64) {}
  void recordRigidTriangleSurfaceContactTaskSubmissionWallTime(PxU64) {}
  void recordRigidTriangleSurfaceContactPostSubmitWaitWallTime(PxU64) {}
  void recordRigidTriangleSurfaceContactTaskLeafWallTimes(PxU64, PxU64,
                                                           PxU64) {}
  void recordRigidTriangleSurfaceContactFeaturePlanTaskLeafWallTimes(
      PxU64, PxU64, PxU64, PxU64) {}
  void recordRigidTriangleSurfaceContactFeatureSweptSubstageWallTimes(
      PxU64, PxU64, PxU64, PxU64, PxU64, PxU64) {}
  void recordRigidTriangleSurfaceContactFeatureForwardOwnerQueries(PxU64,
                                                                    PxU64) {}
  void recordRigidTriangleSurfaceContactFeatureForwardOwnerCache(PxU64,
                                                                  PxU64) {}
  void recordRigidTriangleSurfaceContactFeatureDiscreteQueryStats(
      PxU64, PxU64, PxU64, PxU64, PxU64, PxU64, PxU64, PxU64) {}
  void recordRigidTriangleSurfaceContactFeatureRoundRobinTaskFanIn() {}
  void recordRigidTriangleSurfaceContactFeatureRowPrivateOutputTaskFanIn() {}
  void recordSoftPairContactTasksSubmitted(PxU32) {}
  void beginSoftPairContactTask() {}
  void endSoftPairContactTask() {}
  void recordSoftPairContactFanIn() {}
  void recordSerialSoftPairContactFallback() {}
  void recordSelfBvhContactTasksSubmitted(PxU32) {}
  void beginSelfBvhContactTask() {}
  void endSelfBvhContactTask() {}
  void recordSelfBvhContactFanIn() {}
  void recordSerialSelfBvhContactFallback() {}
  void recordBarrierTask() {}
  bool isTaskGraphSerialMode() const { return mTaskGraphSerialMode; }
