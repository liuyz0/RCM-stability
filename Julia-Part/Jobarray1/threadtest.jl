println("total ", Threads.nthreads())
Threads.@threads for i = 1:10
    println(Threads.threadid(), "for ", i)
end