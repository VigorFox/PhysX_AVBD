// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#ifndef PHYSX_SNIPPET_HEADLESS_H
#define PHYSX_SNIPPET_HEADLESS_H

#include "PxPhysicsAPI.h"

#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Snippets {

enum HeadlessExitCode {
  eHEADLESS_PASS = 0,
  eHEADLESS_GATE_FAILED = 1,
  eHEADLESS_CONFIG_ERROR = 2,
  eHEADLESS_UNSUPPORTED = 3
};

enum HeadlessExecution {
  eHEADLESS_PARALLEL,
  eHEADLESS_SEQUENTIAL
};

struct HeadlessOptions {
  bool headless;
  physx::PxSolverType::Enum solverType;
  physx::PxU32 frames;
  physx::PxU32 seed;
  physx::PxU32 dispatcherThreads;
  physx::PxReal dt;
  HeadlessExecution execution;
  bool framesExplicit;
  bool executionExplicit;
  std::string caseName;

  HeadlessOptions()
      : headless(false), solverType(physx::PxSolverType::eAVBD), frames(600),
        seed(1), dispatcherThreads(2), dt(1.0f / 60.0f),
        execution(eHEADLESS_PARALLEL), framesExplicit(false),
        executionExplicit(false), caseName("default") {}
};

class TrackingErrorCallback : public physx::PxErrorCallback {
public:
  TrackingErrorCallback() : mFatalCount(0), mWarningCount(0) {}

  void reset() {
    mFatalCount.store(0, std::memory_order_relaxed);
    mWarningCount.store(0, std::memory_order_relaxed);
  }

  physx::PxU32 getFatalCount() const {
    return mFatalCount.load(std::memory_order_relaxed);
  }

  physx::PxU32 getWarningCount() const {
    return mWarningCount.load(std::memory_order_relaxed);
  }

  virtual void reportError(physx::PxErrorCode::Enum code, const char *message,
                           const char *file, int line) PX_OVERRIDE {
    switch (code) {
    case physx::PxErrorCode::eINVALID_PARAMETER:
    case physx::PxErrorCode::eINVALID_OPERATION:
    case physx::PxErrorCode::eOUT_OF_MEMORY:
    case physx::PxErrorCode::eINTERNAL_ERROR:
    case physx::PxErrorCode::eABORT:
      mFatalCount.fetch_add(1, std::memory_order_relaxed);
      break;
    case physx::PxErrorCode::eDEBUG_WARNING:
    case physx::PxErrorCode::ePERF_WARNING:
      mWarningCount.fetch_add(1, std::memory_order_relaxed);
      break;
    default:
      break;
    }
    mDefault.reportError(code, message, file, line);
  }

private:
  physx::PxDefaultErrorCallback mDefault;
  std::atomic<physx::PxU32> mFatalCount;
  std::atomic<physx::PxU32> mWarningCount;
};

inline bool equalsIgnoreCase(const char *lhs, const char *rhs) {
  if (!lhs || !rhs)
    return false;
  while (*lhs && *rhs) {
    const unsigned char a = static_cast<unsigned char>(*lhs++);
    const unsigned char b = static_cast<unsigned char>(*rhs++);
    if (std::tolower(a) != std::tolower(b))
      return false;
  }
  return *lhs == *rhs;
}

inline bool parseSolverType(const char *value,
                            physx::PxSolverType::Enum &solverType) {
  if (equalsIgnoreCase(value, "pgs")) {
    solverType = physx::PxSolverType::ePGS;
    return true;
  }
  if (equalsIgnoreCase(value, "tgs")) {
    solverType = physx::PxSolverType::eTGS;
    return true;
  }
  if (equalsIgnoreCase(value, "avbd")) {
    solverType = physx::PxSolverType::eAVBD;
    return true;
  }
  return false;
}

inline const char *getSolverTypeName(physx::PxSolverType::Enum solverType) {
  switch (solverType) {
  case physx::PxSolverType::ePGS:
    return "pgs";
  case physx::PxSolverType::eTGS:
    return "tgs";
  case physx::PxSolverType::eAVBD:
    return "avbd";
  default:
    return "unknown";
  }
}

inline const char *getExecutionName(HeadlessExecution execution) {
  return execution == eHEADLESS_SEQUENTIAL ? "sequential" : "parallel";
}

inline bool isEnabledEnvironmentValue(const char *value) {
  return value && value[0] && value[0] != '0';
}

inline bool parseU32(const char *value, physx::PxU32 minimum,
                     physx::PxU32 maximum, physx::PxU32 &result) {
  if (!value || !value[0] || value[0] == '-')
    return false;
  errno = 0;
  char *end = NULL;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno || !end || *end || parsed < minimum || parsed > maximum)
    return false;
  result = static_cast<physx::PxU32>(parsed);
  return true;
}

inline bool parseReal(const char *value, physx::PxReal minimum,
                      physx::PxReal maximum, physx::PxReal &result) {
  if (!value || !value[0])
    return false;
  errno = 0;
  char *end = NULL;
  const double parsed = std::strtod(value, &end);
  if (errno || !end || *end || !std::isfinite(parsed) || parsed < minimum ||
      parsed > maximum)
    return false;
  result = static_cast<physx::PxReal>(parsed);
  return true;
}

inline bool hasOptionPrefix(const char *arg, const char *prefix) {
  return arg && prefix &&
         std::strncmp(arg, prefix, std::strlen(prefix)) == 0;
}

inline bool isCommonHeadlessOption(const char *arg) {
  return arg &&
         (std::strcmp(arg, "--headless") == 0 ||
          hasOptionPrefix(arg, "--solver=") ||
          hasOptionPrefix(arg, "--frames=") ||
          hasOptionPrefix(arg, "--case=") ||
          hasOptionPrefix(arg, "--scenario=") ||
          hasOptionPrefix(arg, "--seed=") ||
          hasOptionPrefix(arg, "--dispatcher-threads=") ||
          hasOptionPrefix(arg, "--workers=") || hasOptionPrefix(arg, "--dt=") ||
          hasOptionPrefix(arg, "--execution="));
}

inline bool parseCommonHeadlessOptions(int argc, const char *const *argv,
                                       const HeadlessOptions &defaults,
                                       HeadlessOptions &options,
                                       std::string &error) {
  options = defaults;
  error.clear();

  options.headless =
      isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS"));
  options.execution = isEnabledEnvironmentValue(
                          std::getenv("PHYSX_AVBD_ITER_DIAG_SEQUENTIAL"))
                          ? eHEADLESS_SEQUENTIAL
                          : eHEADLESS_PARALLEL;

  // Command-line values are authoritative. Detect the two compatibility
  // environment variables that can fail parsing before reading them, so an
  // explicit CLI override is not rejected by stale parent-process state.
  bool solverArgumentPresent = false;
  bool framesArgumentPresent = false;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    solverArgumentPresent = solverArgumentPresent ||
                            hasOptionPrefix(arg, "--solver=");
    framesArgumentPresent = framesArgumentPresent ||
                            hasOptionPrefix(arg, "--frames=");
  }

  const char *solverEnvironment = std::getenv("PHYSX_SNIPPET_SOLVER");
  if (!solverArgumentPresent && solverEnvironment && solverEnvironment[0] &&
      !parseSolverType(solverEnvironment, options.solverType)) {
    error = "invalid PHYSX_SNIPPET_SOLVER";
    return false;
  }

  const char *framesEnvironment = std::getenv("PHYSX_SNIPPET_FRAME_COUNT");
  if (!framesArgumentPresent && framesEnvironment && framesEnvironment[0]) {
    if (!parseU32(framesEnvironment, 1, 100000000u, options.frames)) {
      error = "invalid PHYSX_SNIPPET_FRAME_COUNT";
      return false;
    }
    options.framesExplicit = true;
  }

  bool headlessSeen = false;
  bool solverSeen = false;
  bool framesSeen = false;
  bool caseSeen = false;
  bool seedSeen = false;
  bool dispatcherSeen = false;
  bool dtSeen = false;
  bool executionSeen = false;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!arg)
      continue;

    if (std::strcmp(arg, "--headless") == 0) {
      if (headlessSeen) {
        error = "duplicate --headless";
        return false;
      }
      headlessSeen = true;
      options.headless = true;
    } else if (hasOptionPrefix(arg, "--solver=")) {
      if (solverSeen) {
        error = "duplicate --solver";
        return false;
      }
      solverSeen = true;
      if (!parseSolverType(arg + std::strlen("--solver="),
                           options.solverType)) {
        error = "invalid --solver value";
        return false;
      }
    } else if (hasOptionPrefix(arg, "--frames=")) {
      if (framesSeen) {
        error = "duplicate --frames";
        return false;
      }
      framesSeen = true;
      if (!parseU32(arg + std::strlen("--frames="), 1, 100000000u,
                    options.frames)) {
        error = "invalid --frames value";
        return false;
      }
      options.framesExplicit = true;
    } else if (hasOptionPrefix(arg, "--case=") ||
               hasOptionPrefix(arg, "--scenario=")) {
      if (caseSeen) {
        error = "duplicate --case/--scenario";
        return false;
      }
      caseSeen = true;
      const char *value =
          arg + std::strlen(hasOptionPrefix(arg, "--case=") ? "--case="
                                                             : "--scenario=");
      if (!value[0]) {
        error = "empty --case value";
        return false;
      }
      options.caseName = value;
    } else if (hasOptionPrefix(arg, "--seed=")) {
      if (seedSeen) {
        error = "duplicate --seed";
        return false;
      }
      seedSeen = true;
      if (!parseU32(arg + std::strlen("--seed="), 0, 0xffffffffu,
                    options.seed)) {
        error = "invalid --seed value";
        return false;
      }
    } else if (hasOptionPrefix(arg, "--dispatcher-threads=") ||
               hasOptionPrefix(arg, "--workers=")) {
      if (dispatcherSeen) {
        error = "duplicate dispatcher thread option";
        return false;
      }
      dispatcherSeen = true;
      const char *value =
          arg + std::strlen(hasOptionPrefix(arg, "--dispatcher-threads=")
                                ? "--dispatcher-threads="
                                : "--workers=");
      if (!parseU32(value, 1, 256, options.dispatcherThreads)) {
        error = "invalid dispatcher thread count";
        return false;
      }
    } else if (hasOptionPrefix(arg, "--dt=")) {
      if (dtSeen) {
        error = "duplicate --dt";
        return false;
      }
      dtSeen = true;
      if (!parseReal(arg + std::strlen("--dt="), 1e-6f, 1.0f, options.dt)) {
        error = "invalid --dt value";
        return false;
      }
    } else if (hasOptionPrefix(arg, "--execution=")) {
      if (executionSeen) {
        error = "duplicate --execution";
        return false;
      }
      executionSeen = true;
      const char *value = arg + std::strlen("--execution=");
      if (equalsIgnoreCase(value, "parallel"))
        options.execution = eHEADLESS_PARALLEL;
      else if (equalsIgnoreCase(value, "sequential"))
        options.execution = eHEADLESS_SEQUENTIAL;
      else {
        error = "invalid --execution value";
        return false;
      }
      options.executionExplicit = true;
    }
  }

  return true;
}

inline bool applyExecutionEnvironment(const HeadlessOptions &options) {
  if (!options.executionExplicit)
    return true;
  const char *value =
      options.execution == eHEADLESS_SEQUENTIAL ? "1" : "0";
#if defined(_WIN32)
  return _putenv_s("PHYSX_AVBD_ITER_DIAG_SEQUENTIAL", value) == 0;
#else
  return setenv("PHYSX_AVBD_ITER_DIAG_SEQUENTIAL", value, 1) == 0;
#endif
}

inline void printHeadlessConfig(const char *snippet,
                                const HeadlessOptions &options) {
  std::printf(
      "[AVBD_GATE_CONFIG] schema=1 snippet=%s solver=%s case=%s "
      "execution=%s frames=%u dt=%.9g dispatcherThreads=%u seed=%u\n",
      snippet, getSolverTypeName(options.solverType), options.caseName.c_str(),
      getExecutionName(options.execution), options.frames, double(options.dt),
      options.dispatcherThreads, options.seed);
}

} // namespace Snippets

#endif // PHYSX_SNIPPET_HEADLESS_H
