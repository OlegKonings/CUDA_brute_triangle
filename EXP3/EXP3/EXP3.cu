#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <map>
#include <ctime>
#include <cuda.h>
#include <math_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
using namespace std;

typedef long long ll;

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice

#define THREADS 256
#define NUM_ELEMENTS 500
#define MEGA (1LL<<29)

const int max_x=NUM_ELEMENTS;
const int max_y=NUM_ELEMENTS;

const int blockSize0=2048;

struct three_p{
	int3 a;
	int num;
};

inline int get_adj_size(const long long num_elem){
	double p=double(num_elem)/double(MEGA);
	if(p>0.8)return 5;
	else if(p>0.6)return 4;
	else if(p>0.4)return 3;
	else if(p>0.2)return 2;
	else
		return 1;
}
inline int get_dynamic_block_size(const int adj_size,const int blkSize){
	return (1<<(adj_size-1))*blkSize;//chk
}

bool is_in_triag(const float2 &p1, const float2 &p2, const float2 &p3, const float2 &p);
void generate_random_points(float2 *Arr, const int sz,const int mx);
three_p CPU_version(const float2 *Arr, const int sz);

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

ll choose(int n, int k);

inline int choose2(int n){return n>0 ? ((n*(n-1))>>1):0;}

inline long long choose3_big(int n){
	long long nn=long long(n);
	return ((((nn*(nn-1LL))>>1LL)*(nn-2LL))/3LL);
}
	//long long nn=long long(n);
	//return ((((nn*(nn-1LL))>>1LL)*(nn-2LL))/3LL);
__device__ __forceinline__ int d_choose2(int n){return n>0 ? ((n*(n-1))>>1):0;}

__device__ __forceinline__ long long d_choose3_big(int n){
	return ((((long long(n)*(long long(n)-1LL))>>1LL)*(long long(n)-2LL))/3LL);
}

__constant__ float2 Pnt_Arr[NUM_ELEMENTS+1];//careful here, __constant__ memory has a 65536 byte limit. 

template<int blockWork>
__global__ void _gpu_optimal_three(int3 *combo,int *best_num,const int sz){

	const long long offset=long long(threadIdx.x)+long long(blockIdx.x)*long long(blockWork);
	const int reps=blockWork>>8;
	const int warpIndex = threadIdx.x%32;

	__shared__ int blk_best[8];
	__shared__ int3 combo_best[8];

	int thread_best=0;
	int3 cur_best;

	float2 p1,p2,p3,p;
	float alpha,beta;
	int ii=0,i,j,k,lo,hi,mid;
	long long pos,cur;

	for(;ii<reps;ii++){
		pos=offset+long long(ii*THREADS);//will be combo number
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=d_choose3_big(mid);
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=d_choose3_big(lo);
		i=lo;
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=long long(d_choose2(mid));
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=long long(d_choose2(lo));
		j=lo;
		k=int(pos);
		// now have idx i,j,k
		p1=Pnt_Arr[i];p2=Pnt_Arr[j];p3=Pnt_Arr[k];
		lo=0;
		for(mid=0;mid<sz;mid++)if(mid!=i && mid!=j && mid!=k){
			p=Pnt_Arr[mid];
			alpha = ((p2.y - p3.y)*(p.x - p3.x) + (p3.x - p2.x)*(p.y - p3.y)) /((p2.y - p3.y)*(p1.x - p3.x) + (p3.x - p2.x)*(p1.y - p3.y));
			beta = ((p3.y - p1.y)*(p.x - p3.x) + (p1.x - p3.x)*(p.y - p3.y)) /((p2.y - p3.y)*(p1.x - p3.x) + (p3.x - p2.x)*(p1.y - p3.y));
			if(alpha>=0.0f && beta>=0.0f && (1.0f - alpha - beta)>=0.0f )lo++;
		}
		if(lo>thread_best){
			thread_best=lo;
			cur_best.x=i;
			cur_best.y=j;
			cur_best.z=k;
		}
	}

	for(ii=16;ii>0;ii>>=1){
		mid=__shfl(thread_best,warpIndex+ii);
		i=__shfl(cur_best.x,warpIndex+ii);
		j=__shfl(cur_best.y,warpIndex+ii);
		k=__shfl(cur_best.z,warpIndex+ii);
		if(mid>thread_best){
			thread_best=mid;
			cur_best.x=i;
			cur_best.y=j;
			cur_best.z=k;
		}
	}
	if(warpIndex==0){
		blk_best[threadIdx.x>>5]=thread_best;
		combo_best[threadIdx.x>>5]=cur_best;
	}
	__syncthreads();

	if(threadIdx.x==0){

		mid=blk_best[0];
		cur_best=combo_best[0];

		if(blk_best[1]>mid){
			mid=blk_best[1];
			cur_best=combo_best[1];
		}
		if(blk_best[2]>mid){
			mid=blk_best[2];
			cur_best=combo_best[2];
		}
		if(blk_best[3]>mid){
			mid=blk_best[3];
			cur_best=combo_best[3];
		}
		if(blk_best[4]>mid){
			mid=blk_best[4];
			cur_best=combo_best[4];
		}
		if(blk_best[5]>mid){
			mid=blk_best[5];
			cur_best=combo_best[5];
		}
		if(blk_best[6]>mid){
			mid=blk_best[6];
			cur_best=combo_best[6];
		}
		if(blk_best[7]>mid){
			mid=blk_best[7];
			cur_best=combo_best[7];
		}
		best_num[blockIdx.x]=mid;
		combo[blockIdx.x]=cur_best;
	}
}

__global__ void tri_last_step(int3 *combo,int *best_num,const int sz, const long long rem_start,const long long bound,const int num_blox){

	const long long offset=long long(threadIdx.x)+rem_start;
	const int warpIndex = threadIdx.x%32;

	__shared__ int blk_best[8];
	__shared__ int3 combo_best[8];

	int thread_best=0;
	int3 cur_best;

	float2 p1,p2,p3,p;
	float alpha,beta;
	int ii=1,i,j,k,lo,hi,mid;
	long long pos,cur,adj=0LL;
	for(;(offset+adj)<bound;ii++){
		pos=offset+adj;
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=d_choose3_big(mid);
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=d_choose3_big(lo);
		i=lo;
		lo=0;hi=sz+1;
		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=long long(d_choose2(mid));
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=long long(d_choose2(lo));
		j=lo;
		k=int(pos);
		// now have idx i,j,k
		p1=Pnt_Arr[i];p2=Pnt_Arr[j];p3=Pnt_Arr[k];
		lo=0;
		for(mid=0;mid<sz;mid++)if(mid!=i && mid!=j && mid!=k){
			p=Pnt_Arr[mid];
			alpha = ((p2.y - p3.y)*(p.x - p3.x) + (p3.x - p2.x)*(p.y - p3.y)) /((p2.y - p3.y)*(p1.x - p3.x) + (p3.x - p2.x)*(p1.y - p3.y));
			beta = ((p3.y - p1.y)*(p.x - p3.x) + (p1.x - p3.x)*(p.y - p3.y)) /((p2.y - p3.y)*(p1.x - p3.x) + (p3.x - p2.x)*(p1.y - p3.y));
			if(alpha>=0.0f && beta>=0.0f && (1.0f - alpha - beta)>=0.0f )lo++;
		}
		if(lo>thread_best){
			thread_best=lo;
			cur_best.x=i;
			cur_best.y=j;
			cur_best.z=k;
		}
		adj=(long long(ii)<<8LL);
	}

	adj=0LL;
	for(ii=1;(threadIdx.x+int(adj))<num_blox;ii++){
		mid=(threadIdx.x+int(adj));
		if(best_num[mid]>thread_best){
			thread_best=best_num[mid];
			cur_best=combo[mid];
		}
		adj=(long long(ii)<<8LL);
	}

	for(ii=16;ii>0;ii>>=1){
		mid=__shfl(thread_best,warpIndex+ii);
		i=__shfl(cur_best.x,warpIndex+ii);
		j=__shfl(cur_best.y,warpIndex+ii);
		k=__shfl(cur_best.z,warpIndex+ii);
		if(mid>thread_best){
			thread_best=mid;
			cur_best.x=i;
			cur_best.y=j;
			cur_best.z=k;
		}
	}

	if(warpIndex==0){
		blk_best[threadIdx.x>>5]=thread_best;
		combo_best[threadIdx.x>>5]=cur_best;
	}
	__syncthreads();

	if(threadIdx.x==0){
		mid=blk_best[0];
		cur_best=combo_best[0];
		if(blk_best[1]>mid){
			mid=blk_best[1];
			cur_best=combo_best[1];
		}
		if(blk_best[2]>mid){
			mid=blk_best[2];
			cur_best=combo_best[2];
		}
		if(blk_best[3]>mid){
			mid=blk_best[3];
			cur_best=combo_best[3];
		}
		if(blk_best[4]>mid){
			mid=blk_best[4];
			cur_best=combo_best[4];
		}
		if(blk_best[5]>mid){
			mid=blk_best[5];
			cur_best=combo_best[5];
		}
		if(blk_best[6]>mid){
			mid=blk_best[6];
			cur_best=combo_best[6];
		}
		if(blk_best[7]>mid){
			mid=blk_best[7];
			cur_best=combo_best[7];
		}
		best_num[0]=mid;
		combo[0]=cur_best;
	}

}

int main(){
	char ch;
	srand(time(NULL));
	
	const int num_points=NUM_ELEMENTS;
	const long long range=choose3_big(num_points);
	cout<<"\nExpected iterations CPU= "<<range*long long(num_points)<<'\n';
	cout<<"\nExpected iterations GPU= "<<range*long long(11)*long long(num_points)<<'\n';
	const int num_bytes_arr=num_points*sizeof(float2);
	float2 *CPU_Arr=(float2 *)malloc(num_bytes_arr);
	generate_random_points(CPU_Arr,num_points,max_x);
		
	three_p CPU_ans={0},GPU_ans={0};
	cudaError_t err=cudaDeviceReset();
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	cout<<"\nRunning CPU implementation..\n";
    UINT wTimerRes = 0;
	DWORD CPU_time=0,GPU_time=0;
    bool init = InitMMTimer(wTimerRes);
    DWORD startTime=timeGetTime();

	CPU_ans=CPU_version(CPU_Arr,num_points);

	DWORD endTime = timeGetTime();
    CPU_time=endTime-startTime;

    cout<<"CPU solution timing: "<<CPU_time<<'\n';
	cout<<"CPU best value= "<<CPU_ans.num<<" , point indexes ( "<<CPU_ans.a.x<<" , "<<CPU_ans.a.y<<" , "<<CPU_ans.a.z<<" ).\n";

    DestroyMMTimer(wTimerRes, init);

	err=cudaMemcpyToSymbol(Pnt_Arr,CPU_Arr,num_bytes_arr);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	
	const int adj_size=get_adj_size(range);
	const int temp_blocks_sz=get_dynamic_block_size(adj_size,blockSize0);
	const int num_blx=int(range/long long(temp_blocks_sz));
	const long long rem_start=range-(range-long long(num_blx)*long long(temp_blocks_sz));

	//float2 *GPU_Arr;
	int *GPU_best;
	int3 *GPU_combo;

	//err=cudaMalloc((void**)&GPU_Arr,num_bytes_arr);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void**)&GPU_best,num_blx*sizeof(int));
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void**)&GPU_combo,num_blx*sizeof(int3));
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	wTimerRes = 0;
    init = InitMMTimer(wTimerRes);
    startTime = timeGetTime();

	//err=cudaMemcpy(GPU_Arr,CPU_Arr,num_bytes_arr,_HTD);
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	if(adj_size==1){
		_gpu_optimal_three<blockSize0><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);			
	}else if(adj_size==2){
		_gpu_optimal_three<blockSize0*2><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);
	}else if(adj_size==3){
		_gpu_optimal_three<blockSize0*4><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);
	}else if(adj_size==4){
		_gpu_optimal_three<blockSize0*8><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);
	}else{
		_gpu_optimal_three<blockSize0*16><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);
	}
	err = cudaThreadSynchronize();
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	tri_last_step<<<1,THREADS>>>(GPU_combo,GPU_best,num_points,rem_start,range,num_blx);
	err = cudaThreadSynchronize();
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	err=cudaMemcpy(&GPU_ans.num,GPU_best,sizeof(int),_DTH);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	err=cudaMemcpy(&GPU_ans.a,GPU_combo,sizeof(int3),_DTH);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	endTime = timeGetTime();
    GPU_time=endTime-startTime;
	DestroyMMTimer(wTimerRes, init);

	
	err=cudaFree(GPU_best);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(GPU_combo);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	cout<<"CUDA timing: "<<GPU_time<<'\n';
	cout<<"GPU best value= "<<GPU_ans.num<<" , point indexes ( "<<GPU_ans.a.x<<" , "<<GPU_ans.a.y<<" , "<<GPU_ans.a.z<<" ).\n";
	cout<<"\nNote: If there is more than one triangle which has the same optimal value, the GPU version will return a valid triangle, but not necessarily the first encountered.\n";
	if(GPU_ans.num==CPU_ans.num){
		cout<<"\nSuccess. GPU value matches CPU results!. GPU was "<<double(CPU_time)/double(GPU_time)<<" faster that 3.9 ghz CPU.\n";
	}else{
		cout<<"\nError in calculation!\n";
	}


	free(CPU_Arr);
	//cin>>ch;
	return 0;
}

bool InitMMTimer(UINT wTimerRes){
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 
	return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init){
	if(init)
		timeEndPeriod(wTimerRes);
}

ll choose(int n, int k) {
	if((n==0 && k==0)|| (k==0 || n==k))return 1LL;
	if(k>n || n==0 || n<0 || k<0)return 0LL;
    ll res=1LL; 
    for(ll i=1LL,j=(ll)n; i<=(ll)k;++i,--j){res*=j;res/=i;} 
    return res; 
}

bool is_in_triag(const float2 &p1, const float2 &p2, const float2 &p3, const float2 &p){
	float alpha = ((p2.y - p3.y)*(p.x - p3.x) + (p3.x - p2.x)*(p.y - p3.y)) /
        ((p2.y - p3.y)*(p1.x - p3.x) + (p3.x - p2.x)*(p1.y - p3.y));
	float beta = ((p3.y - p1.y)*(p.x - p3.x) + (p1.x - p3.x)*(p.y - p3.y)) /
		   ((p2.y - p3.y)*(p1.x - p3.x) + (p3.x - p2.x)*(p1.y - p3.y));
	float gamma = 1.0f - alpha - beta;

	return alpha>=0.0f && beta>=0.0f && gamma>=0.0f;
}

void generate_random_points(float2 *Arr, const int sz,const int mx){
	bool *B=(bool *)malloc(mx*mx*sizeof(bool));
	memset(B,false,mx*mx*sizeof(bool));
	int a,b;
	for(int i=0;i<sz;i++){
		do{
			a=rand()%mx;
			b=rand()%mx;
		}while(B[a*mx+b]);

		Arr[i].x=float(a);
		Arr[i].y=float(b);
		B[a*mx+b]=true;

	}
	free(B);
}

three_p CPU_version(const float2 *Arr, const int sz){
	three_p ret={0};
	float2 p1,p2,p3,p;
	for(int i=0;i<sz;i++)for(int j=0;j<i;j++)for(int k=0;k<j;k++){
		p1=Arr[i];p2=Arr[j];p3=Arr[k];
		int c=0;
		for(int m=0;m<sz;m++)if(m!=i && m!=j && m!=k){
			p=Arr[m];
			if(is_in_triag(p1,p2,p3,p))c++;
		}
		if(c>ret.num){
			ret.num=c;
			ret.a.x=i;
			ret.a.y=j;
			ret.a.z=k;
		}
	}


	return ret;
}