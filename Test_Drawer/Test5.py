import concurrent.futures as cf
import multiprocessing as mp

def square(n):
    return n,n**2





if __name__ == '__main__':
    slist = []
    
    multiproc = []
    with cf.ProcessPoolExecutor(max_workers=6) as executor:
        for j in range(20):
            fut = executor.submit(square,j)
            multiproc.append(fut)
        for f in cf.as_completed(multiproc):
            a,b = f.result()
            slist.append([a,b])
    print(slist)

