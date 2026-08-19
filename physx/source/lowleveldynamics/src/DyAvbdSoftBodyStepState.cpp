// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

// Keep the Scene-only P4/P5 continuation implementation out of every
// translation unit that consumes the scalar component API. The declaration
// and minimal shared particle-primal contract remain in the component header;
// this TU is the sole owner of state-machine method definitions.
#define DY_AVBD_SOFT_BODY_STEP_STATE_IMPLEMENTATION
#include "DyAvbdSoftBodyComponent.h"
