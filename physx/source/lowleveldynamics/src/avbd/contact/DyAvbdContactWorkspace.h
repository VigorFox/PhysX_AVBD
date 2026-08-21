// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

#ifndef DY_AVBD_CONTACT_WORKSPACE_H
#define DY_AVBD_CONTACT_WORKSPACE_H

#include "avbd/contact/DyAvbdContact.h"
#include "foundation/PxArray.h"

namespace physx
{
namespace Dy
{

// Epoch-owned storage for contact state transfer. It is intentionally small:
// geometric detectors may reuse it without owning solver scratch or pair
// state. The arrays are only valid between begin/copy and reset of one epoch.
struct AvbdContactEpochWorkspace
{
	PxArray<AvbdSoftContact> previousContacts;
	PxArray<PxU8> previousUsed;

	void reserve(PxU32 contactCapacity)
	{
		previousContacts.reserve(contactCapacity);
		previousUsed.reserve(contactCapacity);
	}

	void reset()
	{
		previousContacts.reset();
		previousUsed.reset();
	}
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_WORKSPACE_H
