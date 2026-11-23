#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <vector>

struct AvbdProbeStepTimer
{
    using Clock=std::chrono::steady_clock;
    std::vector<double> times;
    const double cpuStart=cpuSeconds();
    Clock::time_point start;
    static double cpuSeconds()
    {
        FILETIME creation,exit,kernel,user;
        GetProcessTimes(GetCurrentProcess(),&creation,&exit,&kernel,&user);
        ULARGE_INTEGER k,u;k.LowPart=kernel.dwLowDateTime;k.HighPart=kernel.dwHighDateTime;
        u.LowPart=user.dwLowDateTime;u.HighPart=user.dwHighDateTime;
        return double(k.QuadPart+u.QuadPart)*1.e-7;
    }
    void begin(){start=Clock::now();}
    void end(){times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());}
    void summarize(double& mean,double& p95,double& cpuMs)
    {
        const auto n=times.size();
        mean=p95=cpuMs=0;if(!n)return;
        cpuMs=(cpuSeconds()-cpuStart)*1000/n;
        for(double ms:times)mean+=ms/n;
        std::sort(times.begin(),times.end());p95=times[std::min(n-1,size_t(.95*n))];
    }
};
