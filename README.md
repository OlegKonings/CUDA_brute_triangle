CUDA_brute_triangle
===================

brute force examines all n choose k triangles

This code goes through every possible 3 point combination of a set of points, makes a triangle of those three points, then evalutes how many other points are within that triangle. 

In this simple example case, the objective is to find which triangle of the possible set contains within the greatest number of other points. The two different CPU and GPU functions return that max number of points, and the INDEXES of the three points which created that optimal triangle.

While many CUDA GPU implementations of algorithms many only be 10-100 times faster than a single core CPU implementation, this problem has a much greater difference in performance.

The larger the data set of points, the greater the outperformance of the CUDA GPU implementation. For data sets of points >=400 the GPU CUDA implementation was at least 1000x times faster than a 3.9 GHz CPU implementation(including all host-device, device-device and device-host memory copies). 

Python,Ruby, Java or C# fans, please post your times for the equivalent task. I fail to see why anybody uses such slow verbose languages, but you are welcome to prove me wrong(but prove with code and examples).

Optimal Triangle Running Time comparison:
---
<table>
<tr>
    <th>Number of points</th><th>Intel I-3770K 3.9 Ghz CPU time </th><th>Tesla K20c GPU time </th><th> CUDA Speedup</th>
</tr>
    <tr>
    <td> 300</td><td> 35,964 ms </td><td> 36 ms </td><td> 999x</td>
  </tr
  <tr>
    <td> 400</td><td> 114,359 ms </td><td> 110 ms </td><td> 1039.63x </td>
</tr>
<tr>
    <td> 500</td><td> 281,730 ms</td><td> 240 ms </td><td> 1173.875x </td>
</tr>
<tr>
    <td> 700</td><td> 1,095,055 ms</td><td> 998 ms </td><td> 1097.24x </td>
</tr>
<tr>
    <td> 1300</td><td> 13,169,820 ms</td><td> 11672 ms </td><td> 1128.32x </td>
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

