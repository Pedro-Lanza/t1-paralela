#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

enum { SEND_SYNC=0, SEND_ISEND, SEND_RSEND, SEND_BSEND, SEND_SSEND };
enum { RECV_SYNC=0, RECV_IRECV };

static int isPrime(int n){
    if(n<2) return 0;
    int limit = (int)sqrt(n);
    for(int i=2;i<=limit;++i) if(n%i==0) return 0;
    return 1;
}

static void sendInt(int *buf,int dst,int tag,int mode){
    MPI_Request req;
    switch(mode){
        case SEND_SYNC:  MPI_Send(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD); break;
        case SEND_ISEND: MPI_Isend(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD,&req);
                         MPI_Wait(&req,MPI_STATUS_IGNORE); break;
        case SEND_RSEND: MPI_Rsend(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
        case SEND_BSEND: MPI_Bsend(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
        case SEND_SSEND: MPI_Ssend(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
    }
}

static void recvInt(int *buf,int src,int tag,int mode){
    MPI_Request req;
    switch(mode){
        case RECV_SYNC:  MPI_Recv(buf,1,MPI_INT,src,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); break;
        case RECV_IRECV: MPI_Irecv(buf,1,MPI_INT,src,tag,MPI_COMM_WORLD,&req);
                         MPI_Wait(&req,MPI_STATUS_IGNORE); break;
    }
}

static void sendRange(int *range,int dst,int tag,int mode){
    MPI_Request req;
    switch(mode){
        case SEND_SYNC:  MPI_Send(range,2,MPI_INT,dst,tag,MPI_COMM_WORLD); break;
        case SEND_ISEND: MPI_Isend(range,2,MPI_INT,dst,tag,MPI_COMM_WORLD,&req);
                         MPI_Wait(&req,MPI_STATUS_IGNORE); break;
        case SEND_RSEND: MPI_Rsend(range,2,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
        case SEND_BSEND: MPI_Bsend(range,2,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
        case SEND_SSEND: MPI_Ssend(range,2,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
    }
}

static void recvRange(int *range,int src,int tag,int mode){
    MPI_Request req;
    switch(mode){
        case RECV_SYNC:  MPI_Recv(range,2,MPI_INT,src,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); break;
        case RECV_IRECV: MPI_Irecv(range,2,MPI_INT,src,tag,MPI_COMM_WORLD,&req);
                         MPI_Wait(&req,MPI_STATUS_IGNORE); break;
    }
}

int main(int argc,char **argv){
    MPI_Init(&argc,&argv);

    int processRank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD,&processRank);
    MPI_Comm_size(MPI_COMM_WORLD,&processCount);

    if(argc<3){
        if(processRank==0)
            printf("Uso: %s <modo_send 0–4> <modo_recv 0–1>\n",argv[0]);
        MPI_Finalize();
        return 1;
    }

    int sendMode = atoi(argv[1]);
    int recvMode = atoi(argv[2]);
    int N = 100000, base = N/processCount;
    int range[2], partialCount=0, totalCount=0;
    double t0, t1, elapsedMaster, elapsedTotal;

    if(processRank==0){
        t0 = MPI_Wtime();
        for(int dst=1; dst<processCount; ++dst){
            range[0] = dst*base;
            range[1] = (dst==processCount-1 ? N : range[0]+base);
            sendRange(range,dst,0,sendMode);
        }
        // master does its own slice
        range[0] = 0;
        range[1] = base;
        for(int i=range[0]; i<range[1]; ++i)
            if(isPrime(i)) totalCount++;

        for(int src=1; src<processCount; ++src){
            recvInt(&partialCount,src,1,recvMode);
            totalCount += partialCount;
        }
        t1 = MPI_Wtime();
        elapsedMaster = t1 - t0;

        printf("Total primes=%d  Tempo=%.6fs\n", totalCount, elapsedMaster);
    } else {
        recvRange(range,0,0,recvMode);
        for(int i=range[0]; i<range[1]; ++i)
            if(isPrime(i)) partialCount++;
        sendInt(&partialCount,0,1,sendMode);
    }

    MPI_Finalize();
    return 0;
}
