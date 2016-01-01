CUDA_brute_triangle
===================

brute force examines all possible n choose 3 triangles only once

UPDATE: Optimized both implementations, so new times posted.Run using the  --use_fast_math flag, and must have a GPU with compute capability >=3.5. Best performance is max_register set to 32.

This code goes through every possible 3 point combination of a set of points, makes a triangle of those three points, then evalutes how many other points are within that triangle. 

In this simple example case, the objective is to find which triangle of the possible set contains within its borders the greatest number of other points(exclusive). The two different CPU and GPU functions return that max number of points, and the INDEXES of the three points which created that optimal triangle.

NOTE: Since the GPU version does not proceed in a serial fashion, if there is more than one combination associated with the optimal value it will return a valid combination, BUT not necessarly the same(first seen) combination the CPU version returned.  The overall answer will be correct, but a hueristic needs to be added for the CUDA version to return a specific arrangement if more than one does indeed exist.

While many CUDA GPU implementations of algorithms many only be 10-100 times faster than a single core CPU implementation, this problem has a much greater difference in performance.

The larger the data set of points, the greater the outperformance of the CUDA GPU implementation. For data sets of points >=400 the GPU CUDA implementation was at least 400x times faster than a 3.9 Ghz CPU implementation(including all host-device, device-device and device-host memory copies). 

No overlocking of GPU, is running at stock 706 Mhz for older Kepler Tesla K20c

Optimal Triangle Running Time comparison:
---
<table>
<tr>
    <th>Number of points</th><th>Intel I-3770K 3.9 Ghz CPU time </th><th>Tesla K20c GPU time </th><th> CUDA Speedup</th>
</tr>
    <tr>
    <td> 300</td><td> 13,198 ms </td><td> 32 ms </td><td> 412.438x</td>
  </tr
  <tr>
    <td> 400</td><td> 42,710 ms </td><td> 99 ms </td><td> 431.32x </td>
</tr>
<tr>
    <td> 500</td><td> 103,731 ms</td><td> 240 ms </td><td> 432.2x </td>
</tr>
<tr>
    <td> 700</td><td> 411,401 ms</td><td> 912 ms </td><td> 451.01x </td>
</tr>
<tr>
    <td> 1300</td><td> 5,075,524 ms</td><td> 10,788 ms </td><td> 470.47x </td>
</tr>
</table>


New Times for GTX 980 and GTX 780ti

<table>
<tr>
    <th>Number of points</th><th>Intel I-3770K 3.9 Ghz CPU time </th><th>GTX 980 time </th><th>GTX 780ti time </th><th>GTX Titan X time </th>
</tr>
    
<tr>
    <td> 700</td><td> 411,401 ms</td><td> 716 ms </td><td> 655 ms </td><td> 573 ms </td>
</tr>
<tr>
    <td> 1300</td><td> 5,075,524 ms</td><td> 8,422 ms </td><td> 7,720 ms </td><td> 6,735 ms </td>
</tr>
</table>

___

The running time is apx (N choose 3)*N, where N is the number of 2-D points in the array. 

What makes this even more impressive is that the GPU actually has to do over 10x more work for the same answer (this is apparent in code, compare CPU version to GPU).


AMD users, post your times and code. I would like to compare results. Same goes to the multi-core CPU crowd who thinks a Xeon can beat a GPU at this type of problem. Prove it ..

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-60172288-1', 'auto');
  ga('send', 'pageview');

</script>



