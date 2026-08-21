// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		void AvbdCpuSoftScene::prepareIslandGeneration(
			PxReal dt, const PxVec3& gravity, bool sleepingEnabled)
		{
			if(mEntries.empty() || mParticles.empty())
				return;

			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Dy::DeformableBodyCore& core =
					mEntries[i].getBodyCore();
				if(!sleepingEnabled ||
					core.cpuAvbdWakeRequested)
				{
					const PxReal wakeCounter =
						core.wakeCounter > 0.0f
							? core.wakeCounter
							: ScInternalWakeCounterResetValue;
					wakeEntry(
						mEntries[i], wakeCounter);
				}
				syncHostInputs(
					mEntries[i], mDeformableMaterialManager);
			}
			refreshVolumeKinematicTargets();
			refreshPrescribedAttachmentTargets();

			for(PxU32 i = 0; i < mNativeIslandEdges.size(); i++)
				mNativeIslandEdges[i].touched = false;
			for(PxU32 i = 0;
				i < mNativeSoftSoftIslandEdges.size(); i++)
				mNativeSoftSoftIslandEdges[i].touched = false;

			// Public rigid attachments are persistent island topology, unlike
			// proximity contacts. Keep their native edge alive even when the
			// attached actors have no overlapping simulation shapes.
			for(PxU32 i = 0; i < mRigidAttachments.size(); i++)
			{
				RigidAttachmentEntry& attachment =
					mRigidAttachments[i];
				Entry* softEntry =
					findEntry(*attachment.softCore);
				BodySim* bodySim =
					attachment.rigidCore->getSim();
				if(!softEntry || !bodySim ||
					bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;
				if(softEntry->sleeping && bodySim->isActive())
					wakeEntry(
						*softEntry,
						ScInternalWakeCounterResetValue);
				else if(!softEntry->sleeping &&
					!bodySim->isActive())
					attachment.rigidCore->wakeUp(
						ScInternalWakeCounterResetValue);
				ensureNativeIslandEdge(
					*softEntry, *attachment.rigidCore);
			}

			// Articulation-link attachments are persistent topology too, but
			// their solve owner is a generalized-coordinate position block.
			for(PxU32 i = 0;
				i < mArticulationAttachments.size(); i++)
			{
				ArticulationAttachmentEntry& attachment =
					mArticulationAttachments[i];
				Entry* softEntry =
					findEntry(*attachment.softCore);
				BodySim* bodySim =
					attachment.linkCore->getSim();
				if(!softEntry || !bodySim ||
					!bodySim->isArticulationLink() ||
					!bodySim->getArticulation())
					continue;
				if(softEntry->sleeping && bodySim->isActive())
					wakeEntry(
						*softEntry,
						ScInternalWakeCounterResetValue);
				else if(!softEntry->sleeping &&
					!bodySim->isActive())
					attachment.linkCore->wakeUp(
						ScInternalWakeCounterResetValue);
				ensureNativeIslandEdge(
					*softEntry, *attachment.linkCore);
			}

			// A public deformable-pair attachment is persistent island
			// topology. It must keep both soft actors in one selection even
			// after their collision bounds separate.
			for(PxU32 i = 0; i < mSoftPairAttachments.size(); i++)
			{
				SoftPairAttachmentEntry& attachment =
					mSoftPairAttachments[i];
				Entry* softEntry0 =
					findEntry(*attachment.softCore[0]);
				Entry* softEntry1 =
					findEntry(*attachment.softCore[1]);
				if(!softEntry0 || !softEntry1)
					continue;
				if(softEntry0->sleeping && !softEntry1->sleeping)
					wakeEntry(
						*softEntry0,
						ScInternalWakeCounterResetValue);
				else if(softEntry1->sleeping &&
					!softEntry0->sleeping)
					wakeEntry(
						*softEntry1,
						ScInternalWakeCounterResetValue);
				ensureNativeSoftSoftIslandEdge(
					*softEntry0, *softEntry1);
			}

			for(PxU32 softIndex = 0;
				softIndex < mEntries.size(); softIndex++)
			{
				Entry& softEntry = mEntries[softIndex];
				PxBounds3 softBounds;
				if(!computeCollisionDomainSoftBounds(softEntry, softBounds))
					continue;
				const bool speculativeCCDEnabled =
					softEntry.bodyIndex < mBodies.size() &&
					mBodies[softEntry.bodyIndex].compiled.
						speculativeCCDEnabled;
				if(speculativeCCDEnabled)
				{
					if(!expandSoftBoundsForPrediction(
						softEntry, dt, gravity, softBounds))
						continue;
				}
				else if(!computePredictedCollisionDomainSoftBounds(
					softEntry, dt, gravity, softBounds))
				{
					// Non-CCD AVBD uses an endpoint-only DCD admission below.
					// Do not retain the source-pose AABB here: that would turn
					// the topology decision into a swept candidate.
					continue;
				}

				for(PxU32 shapeIndex = 0;
					shapeIndex < mDynamicShapes.size(); shapeIndex++)
				{
					const DynamicShapeEntry& dynamicEntry =
						mDynamicShapes[shapeIndex];
					BodySim* const shapeBodySim =
						dynamicEntry.core
							? dynamicEntry.core->getSim() : NULL;
					Dy::AvbdRigidBox box;
					PxBounds3 rigidBounds;
					if(compileDynamicBox(dynamicEntry, box))
					{
						rigidBounds = computeBoxBounds(box);
						// Keep non-CCD topology admission at one discrete endpoint.
						// A sphere bound is used only for the broad phase so arbitrary
						// rigid rotation cannot make us miss that endpoint; the narrow
						// phase below remains an OBB current-pose query, never a sweep.
						if(!speculativeCCDEnabled && shapeBodySim &&
							!shapeBodySim->isKinematic())
						{
							if(!computeDynamicEndpointEnvelopeBounds(
									dynamicEntry, box.center,
									box.halfExtent.magnitude(), dt, gravity,
									rigidBounds))
								continue;
						}
					}
					else
					{
						Dy::AvbdRigidSphere sphere;
						if(compileDynamicSphere(
								dynamicEntry, sphere))
						{
							rigidBounds = computeSphereBounds(sphere);
							if(!speculativeCCDEnabled && shapeBodySim &&
								!shapeBodySim->isKinematic())
							{
								if(!computeDynamicEndpointEnvelopeBounds(
										dynamicEntry, sphere.center,
										sphere.radius, dt, gravity,
										rigidBounds))
									continue;
							}
							// A dynamic sphere that crosses a soft actor within
							// one frame needs native island topology before the
							// solver-body prediction is available. Bound the
							// current/predicted body-center segment by the sphere
							// radius plus its shape offset; this is conservative
							// for arbitrary rotation and is public-flag gated.
							BodySim* sphereBodySim =
								dynamicEntry.core->getSim();
							if(speculativeCCDEnabled &&
								sphereBodySim &&
								!sphereBodySim->isKinematic())
							{
								const PxsBodyCore& bodyCore =
									dynamicEntry.core->getCore();
								const PxVec3 bodyCenter =
									bodyCore.body2World.p;
								const PxReal shapeOffset =
									(sphere.center - bodyCenter).
										magnitude();
								const PxReal envelopeRadius =
									sphere.radius + shapeOffset;
								const PxVec3 predictedBodyCenter =
									bodyCenter +
									bodyCore.linearVelocity * dt +
									(bodyCore.disableGravity
										? PxVec3(0.0f)
										: gravity * (dt * dt));
								if(!PxIsFinite(shapeOffset) ||
									!PxIsFinite(envelopeRadius) ||
									!predictedBodyCenter.isFinite())
									continue;
								const PxVec3 envelopeExtent(
									envelopeRadius);
								rigidBounds.include(
									bodyCenter - envelopeExtent);
								rigidBounds.include(
									bodyCenter + envelopeExtent);
								rigidBounds.include(
									predictedBodyCenter -
										envelopeExtent);
								rigidBounds.include(
									predictedBodyCenter +
										envelopeExtent);
							}
						}
						else
						{
							Dy::AvbdRigidCapsule capsule;
							if(compileDynamicCapsule(
									dynamicEntry, capsule))
							{
								rigidBounds =
									computeCapsuleBounds(capsule);
								if(!speculativeCCDEnabled && shapeBodySim &&
									!shapeBodySim->isKinematic())
								{
									if(!computeDynamicEndpointEnvelopeBounds(
											dynamicEntry, capsule.center,
											capsule.radius + capsule.halfHeight,
											dt, gravity, rigidBounds))
										continue;
								}
								BodySim* capsuleBodySim =
									dynamicEntry.core->getSim();
								if(speculativeCCDEnabled &&
									capsuleBodySim)
								{
									if(capsuleBodySim->isKinematic())
									{
										Dy::AvbdRigidCapsule
											previousCapsule = capsule;
										previousCapsule.center =
											capsule.previousCenter;
										previousCapsule.rotation =
											capsule.previousRotation;
										const PxBounds3 previousBounds =
											computeCapsuleBounds(
												previousCapsule);
										rigidBounds.include(
											previousBounds.minimum);
										rigidBounds.include(
											previousBounds.maximum);
										if(!Dy::
											avbdAreSweepRotationsEquivalent(
												capsule.
													previousRotation,
												capsule.rotation))
										{
											// Endpoint AABBs do not contain
											// the arc swept by a rotating
											// capsule. A center-segment
											// sphere envelope is conservative
											// for every intermediate
											// orientation and is only enabled
											// for a speculative source.
											const PxVec3 rotationExtent(
												capsule.radius +
													capsule.halfHeight);
											rigidBounds.include(
												capsule.previousCenter -
													rotationExtent);
											rigidBounds.include(
												capsule.previousCenter +
													rotationExtent);
											rigidBounds.include(
												capsule.center -
													rotationExtent);
											rigidBounds.include(
												capsule.center +
													rotationExtent);
										}
									}
									else
									{
										const PxsBodyCore& bodyCore =
											dynamicEntry.core->getCore();
										const PxVec3 bodyCenter =
											bodyCore.body2World.p;
										const PxReal shapeOffset =
											(capsule.center -
												bodyCenter).magnitude();
										const PxReal envelopeRadius =
											capsule.radius +
											capsule.halfHeight +
											shapeOffset;
										const PxVec3 predictedBodyCenter =
											bodyCenter +
											bodyCore.linearVelocity * dt +
											(bodyCore.disableGravity
												? PxVec3(0.0f)
												: gravity * (dt * dt));
										if(!PxIsFinite(shapeOffset) ||
											!PxIsFinite(
												envelopeRadius) ||
											!predictedBodyCenter.isFinite())
											continue;
										const PxVec3 envelopeExtent(
											envelopeRadius);
										rigidBounds.include(
											bodyCenter - envelopeExtent);
										rigidBounds.include(
											bodyCenter + envelopeExtent);
										rigidBounds.include(
											predictedBodyCenter -
												envelopeExtent);
										rigidBounds.include(
											predictedBodyCenter +
												envelopeExtent);
									}
								}
							}
							else
							{
								Dy::AvbdRigidConvex convex;
								if(compileDynamicConvex(
										dynamicEntry, convex))
								{
									rigidBounds =
										computeConvexBounds(convex);
									if(!speculativeCCDEnabled && shapeBodySim &&
										!shapeBodySim->isKinematic())
									{
										if(!computeDynamicEndpointEnvelopeBounds(
												dynamicEntry, convex.center,
												convex.localRadius, dt, gravity,
												rigidBounds))
											continue;
									}
									BodySim* convexBodySim =
										dynamicEntry.core->getSim();
									if(speculativeCCDEnabled &&
										convexBodySim)
									{
										if(convexBodySim->isKinematic())
										{
											Dy::AvbdRigidConvex
												previousConvex = convex;
											previousConvex.center =
												convex.previousCenter;
											previousConvex.rotation =
												convex.previousRotation;
											const PxBounds3 previousBounds =
												computeConvexBounds(
													previousConvex);
											rigidBounds.include(
												previousBounds.minimum);
											rigidBounds.include(
												previousBounds.maximum);
											if(!Dy::
												avbdAreSweepRotationsEquivalent(
													convex.
														previousRotation,
													convex.rotation))
											{
												// The convex is contained by
												// a shape-center sphere with
												// localRadius for every
												// intermediate orientation.
												const PxVec3 rotationExtent(
													convex.localRadius);
												rigidBounds.include(
													convex.previousCenter -
														rotationExtent);
												rigidBounds.include(
													convex.previousCenter +
														rotationExtent);
												rigidBounds.include(
													convex.center -
														rotationExtent);
												rigidBounds.include(
													convex.center +
														rotationExtent);
											}
										}
										else
										{
											const PxsBodyCore& bodyCore =
												dynamicEntry.core->getCore();
											const PxVec3 bodyCenter =
												bodyCore.body2World.p;
											const PxReal shapeOffset =
												(convex.center -
													bodyCenter).magnitude();
											const PxReal envelopeRadius =
												convex.localRadius +
												shapeOffset;
											const PxVec3
												predictedBodyCenter =
													bodyCenter +
													bodyCore.linearVelocity *
														dt +
													(bodyCore.disableGravity
														? PxVec3(0.0f)
														: gravity *
															(dt * dt));
											if(!PxIsFinite(shapeOffset) ||
												!PxIsFinite(
													envelopeRadius) ||
												!predictedBodyCenter.
													isFinite())
												continue;
											const PxVec3 envelopeExtent(
												envelopeRadius);
											rigidBounds.include(
												bodyCenter -
													envelopeExtent);
											rigidBounds.include(
												bodyCenter +
													envelopeExtent);
											rigidBounds.include(
												predictedBodyCenter -
													envelopeExtent);
											rigidBounds.include(
												predictedBodyCenter +
													envelopeExtent);
										}
									}
								}
								else
								{
									Dy::AvbdRigidTriangleSurface
										triangleSurface;
									if(!compileDynamicTriangleSurface(
											dynamicEntry,
											triangleSurface))
										continue;
									rigidBounds =
										computeTriangleSurfaceBounds(
											triangleSurface);
									BodySim* triangleSurfaceBodySim =
										dynamicEntry.core->getSim();
									if(speculativeCCDEnabled &&
										triangleSurfaceBodySim &&
										triangleSurfaceBodySim->
											isKinematic())
									{
										Dy::AvbdRigidTriangleSurface
											previousSurface =
												triangleSurface;
										previousSurface.center =
											triangleSurface.
												previousCenter;
										previousSurface.rotation =
											triangleSurface.
												previousRotation;
										const PxBounds3 previousBounds =
											computeTriangleSurfaceBounds(
												previousSurface);
										rigidBounds.include(
											previousBounds.minimum);
										rigidBounds.include(
											previousBounds.maximum);
										if(!Dy::
											avbdAreSweepRotationsEquivalent(
												triangleSurface.
													previousRotation,
												triangleSurface.rotation))
										{
											// Endpoint AABBs do not contain
											// the arc swept by a rotating
											// triangle surface. Every baked
											// vertex stays inside the
											// shape-center localRadius sphere.
											const PxVec3 rotationExtent(
												triangleSurface.
													localRadius);
											rigidBounds.include(
												triangleSurface.
													previousCenter -
													rotationExtent);
											rigidBounds.include(
												triangleSurface.
													previousCenter +
													rotationExtent);
											rigidBounds.include(
												triangleSurface.center -
													rotationExtent);
											rigidBounds.include(
												triangleSurface.center +
													rotationExtent);
										}
									}
								}
								if(rigidBounds.isEmpty())
									continue;
							}
						}
					}

					PxBounds3 candidateBounds = softBounds;
					const PxReal wakeMargin =
						2.0f * mContactParams.contactRadius +
						PxMax(
							dynamicEntry.shape->getContactOffset(),
							0.0f);
					candidateBounds.fattenSafe(wakeMargin);
					if(!candidateBounds.intersects(rigidBounds))
						continue;

					BodySim* bodySim = shapeBodySim;
					bool rigidNodeActive = false;
					if(bodySim)
					{
						const PxNodeIndex rigidNode =
							bodySim->getNodeIndex();
						const IG::IslandSim& accurateIslandSim =
							mIslandManager.getAccurateIslandSim();
						rigidNodeActive =
							rigidNode.isValid() &&
							rigidNode.index() <
								accurateIslandSim.getNbNodes() &&
							accurateIslandSim.getNode(
								rigidNode).isActive();
					}
					if(softEntry.sleeping && bodySim &&
						(bodySim->isActive() || rigidNodeActive))
					{
						wakeEntry(
							softEntry,
							ScInternalWakeCounterResetValue);
					}
					// Kinematics are prescribed one-way position targets.
					// They wake overlapping soft actors but must not enter
					// the two-sided rigid 6x6 AVBD island objective.
					if(bodySim && !bodySim->isKinematic())
						ensureNativeIslandEdge(
							softEntry, *dynamicEntry.core);
				}
			}

			for(PxU32 softIndex0 = 0;
				softIndex0 < mEntries.size(); softIndex0++)
			{
				Entry& softEntry0 = mEntries[softIndex0];
				PxBounds3 softBounds0;
				if(!computeCollisionDomainSoftBounds(softEntry0, softBounds0))
					continue;
				const PxReal wakeMargin =
					PxMax(mContactParams.contactRadius, 0.0f);
				softBounds0.fattenSafe(wakeMargin);

				for(PxU32 softIndex1 = softIndex0 + 1;
					softIndex1 < mEntries.size(); softIndex1++)
				{
					Entry& softEntry1 = mEntries[softIndex1];
					PxBounds3 softBounds1;
					if(!computeCollisionDomainSoftBounds(softEntry1, softBounds1) ||
						!softBounds0.intersects(softBounds1))
						continue;

					const bool soft0WasSleeping =
						softEntry0.sleeping;
					const bool soft1WasSleeping =
						softEntry1.sleeping;
					if(soft0WasSleeping && !soft1WasSleeping)
					{
						wakeEntry(
							softEntry0,
							ScInternalWakeCounterResetValue);
					}
					else if(soft1WasSleeping && !soft0WasSleeping)
					{
						wakeEntry(
							softEntry1,
							ScInternalWakeCounterResetValue);
					}
					ensureNativeSoftSoftIslandEdge(
						softEntry0, softEntry1);
				}
			}

			for(PxU32 i = mNativeIslandEdges.size();
				i > 0; i--)
			{
				NativeIslandEdgeEntry& edge =
					mNativeIslandEdges[i - 1];
				if(!edge.touched)
				{
					mIslandManager.removeConnection(
						edge.edgeIndex);
					mNativeIslandEdges.replaceWithLast(
						i - 1);
				}
			}
			for(PxU32 i = mNativeSoftSoftIslandEdges.size();
				i > 0; i--)
			{
				NativeSoftSoftIslandEdgeEntry& edge =
					mNativeSoftSoftIslandEdges[i - 1];
				if(!edge.touched)
				{
					mIslandManager.removeConnection(
						edge.edgeIndex);
					mNativeSoftSoftIslandEdges.replaceWithLast(
						i - 1);
				}
			}
		}

} // namespace Sc
} // namespace physx
