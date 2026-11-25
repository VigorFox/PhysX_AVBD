// Late-time observation of the visual snippet's repeated 50 m/s sphere shots.
struct VolleyWindow
{
	double spinSq = 0, slipSq = 0, centerSq = 0, netSq = 0;
	PxReal peakSpin = 0, groundPeakSpin = 0, peakSpinEnergy = 0;
	PxU32 ballSamples = 0, netSamples = 0, contacts = 0, retained = 0;
	PxU32 sleeping = 0, caughtMask = 0;
	bool finite = true, observedRest = true;

	void sample(const Fixture& f, const std::vector<PxRigidDynamic*>& balls)
	{
		retained = sleeping = caughtMask = 0;
		for(PxU32 i=0;i<balls.size();++i)
		{
			const auto* b=balls[i];
			const PxTransform pose=b->getGlobalPose();
			const PxVec3 v=b->getLinearVelocity(), w=b->getAngularVelocity();
			finite &= pose.isFinite() && v.isFinite() && w.isFinite();
			const PxReal spin = w.magnitude();
			if(pose.p.y<=20 || PxAbs(pose.p.x)>SPACING*(W-1)*.5f || PxAbs(pose.p.z)>SPACING*(H-1)*.5f)
			{
				groundPeakSpin=PxMax(groundPeakSpin,spin);
				continue;
			}
			peakSpin = PxMax(peakSpin,spin);
			// All observed projectiles are homogeneous spheres. Express their
			// rotational energy per mass in the SDK sleep-threshold units.
			peakSpinEnergy = PxMax(peakSpinEnergy,.5f*b->getMassSpaceInertiaTensor().x*spin*spin/b->getMass());
			spinSq += double(spin)*spin;
			centerSq += v.magnitudeSquared();
			caughtMask |= 1u<<i;
			sleeping += b->isSleeping() ? 1u : 0u;
			++retained;
			++ballSamples;
		}
		for(const auto* b : f.nodes)
		{
			netSq += b->getLinearVelocity().magnitudeSquared();
			++netSamples;
		}
		const PxU32 previousContacts=contacts;
		for(const auto& w : f.callback.witnesses)
		{
			if(w.point.y<=20)continue;
			const PxVec3 a = w.a ? PxRigidBodyExt::getVelocityAtPos(*w.a,w.point) : PxVec3(0);
			const PxVec3 b = w.b ? PxRigidBodyExt::getVelocityAtPos(*w.b,w.point) : PxVec3(0);
			const PxVec3 v = a-b;
			slipSq += (v-w.normal*v.dot(w.normal)).magnitudeSquared();
			++contacts;
		}
		observedRest &= contacts>previousContacts || (retained && sleeping==retained);
	}
	double rms(double sum,PxU32 count) const {return count ? std::sqrt(sum/count) : 0;}
};

int runVolley(const Snippets::HeadlessOptions& o)
{
	const PxU32 frames=o.frames, observationFrames=300;
	Fixture f;
	if(!f.init(o,true,false))return Snippets::eHEADLESS_GATE_FAILED;
	std::vector<PxRigidDynamic*> balls(1,f.ball);
	VolleyWindow window,quiet;
	PxU32 caughtMask=0;
	bool retainedThroughout=true, finite=true, observedBallPair=false;
	f.callback.observeSlip=true;
	for(PxU32 frame=0;frame<frames;++frame)
	{
		const PxU32 shotCount=24;
		if(frame>=600 && (frame-600)%24==0 && balls.size()<=shotCount)
		{
			const PxU32 i=PxU32(balls.size()-1);
			const PxVec3 target((PxReal(i%5)-2)*1.15f,30,(PxReal((i*7)%11)-5)*.52f);
			const PxVec3 direction=PxVec3(.11f,-1,.09f).getNormalized();
			PxRigidDynamic* shot=PxCreateDynamic(*f.physics,PxTransform(target-direction*16),
				PxSphereGeometry(1),*f.material,100);
			if(!shot)return Snippets::eHEADLESS_GATE_FAILED;
			shot->setLinearVelocity(direction*50.f);
			shot->userData=&gBallTag;
			f.scene->addActor(*shot);
			balls.push_back(shot);
		}
		f.callback.witnesses.clear();
		if(!f.step(o.dt))return Snippets::eHEADLESS_GATE_FAILED;
		window.sample(f,balls);
		finite &= window.finite;
		if(!observedBallPair)
			for(const auto& w:f.callback.witnesses)
				if(w.a && w.b && w.a->userData==&gBallTag && w.b->userData==&gBallTag)observedBallPair=true;
		if(frames-frame<=observationFrames)
		{
			quiet.sample(f,balls);
			if(frames-frame==observationFrames)caughtMask=quiet.caughtMask;
			retainedThroughout &= quiet.caughtMask==caughtMask;
		}
		if((frame+1)%120==0 || frame+1==frames)
		{
			std::printf("[CHAINMAIL_VOLLEY] frame=%u balls=%u retained=%u spinRms=%g peakSpin=%g centerRms=%g netRms=%g slipRms=%g contactSamples=%u groundPeakSpin=%g\n",
				frame+1,PxU32(balls.size()),window.retained,window.rms(window.spinSq,window.ballSamples),
				double(window.peakSpin),window.rms(window.centerSq,window.ballSamples),window.rms(window.netSq,window.netSamples),
				window.rms(window.slipSq,window.contacts),window.contacts,double(window.groundPeakSpin));
			std::fflush(stdout);
			if(frame+1!=frames)window=VolleyWindow();
		}
	}
	const PxReal threshold=f.ball->getSleepThreshold();
	const double slip=quiet.rms(quiet.slipSq,quiet.contacts);
	// The launches can knock balls out of the basket. Observe the same
	// caught majority for the entire final five seconds, including the large ball.
	const bool pass=balls.size()==25 && quiet.retained>balls.size()/2 && (caughtMask&1) &&
		retainedThroughout && quiet.observedRest && observedBallPair && f.metrics.contactPoints && .5*slip*slip<threshold &&
		quiet.peakSpinEnergy<threshold && finite && !f.metrics.nonFinite && !f.errors.getFatalCount();
	std::printf("[AVBD_GATE] snippet=SnippetChainmail case=volley completedFrames=%u status=%s balls=%u retained=%u "
		"steadyCohort=%u sleeping=%u peakSpin=%g spinRms=%g slipRms=%g peakSpinEnergy=%g sleepThreshold=%g\n",
		f.metrics.completed,pass?"PASS":"FAIL",PxU32(balls.size()),quiet.retained,unsigned(retainedThroughout),
		quiet.sleeping,double(quiet.peakSpin),quiet.rms(quiet.spinSq,quiet.ballSamples),slip,double(quiet.peakSpinEnergy),double(threshold));
	return pass?Snippets::eHEADLESS_PASS:Snippets::eHEADLESS_GATE_FAILED;
}
