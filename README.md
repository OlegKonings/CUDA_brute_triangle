CUDA_brute_triangle
===================

brute force examines all possible n choose 3 triangles only once

UPDATE: Optimized both implementations, so new times posted.

This code goes through every possible 3 point combination of a set of points, makes a triangle of those three points, then evalutes how many other points are within that triangle. 

In this simple example case, the objective is to find which triangle of the possible set contains within its borders the greatest number of other points(exclusive). The two different CPU and GPU functions return that max number of points, and the INDEXES of the three points which created that optimal triangle.

While many CUDA GPU implementations of algorithms many only be 10-100 times faster than a single core CPU implementation, this problem has a much greater difference in performance.

The larger the data set of points, the greater the outperformance of the CUDA GPU implementation. For data sets of points >=400 the GPU CUDA implementation was at least 400x times faster than a 3.9 Ghz CPU implementation(including all host-device, device-device and device-host memory copies). 

NOTE: no overlocking of GPU, is running at stock 706 Mhz

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
</table>
___

The running time is apx (N choose 3)*N, where N is the number of 2-D points in the array. 

What makes this even more impressive is that the GPU actually has to do over 10x more work for the same answer (this is apparent in code, compare CPU version to GPU).

Note: Must have compute capability of 3.0 or higher to run(GTX 660 or better).

AMD users, post your times and code. I would like to compare results.

 <script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-43459430-1', 'github.com');
  ga('send', 'pageview');

</script>

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/8024c83bd0328155085f6a67bc179d04 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_brute_triangle)

