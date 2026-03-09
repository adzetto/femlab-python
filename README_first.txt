// The package is still in Beta version.
// It is a transform of the FemLab package created by 
// O.Hededal and S. Krenk, at the Aalborg University.
//
// Translation for scilab was done by G.Turan, at IYTE.
//
// The builder script does not work as it should be. As 
// a workaround a simpler script is doing the task.

// To run an analysis 

1. Open scilab
2. change directory to the FemLab toolbox
3. type the following three lines and you should receive two 
   plots of a cantilever beam.

   exec loader.sce;     // this is the workaround
   exec examples/canti.sce;
   exec examples/elastic.sce;
   
4. Now, investigate the input files for the cantilever beam. 
     a) The file examples/canti.sce contains information
        about the beams geometry, topology, boundary 
        conditions, and external loading.
     b) The file examples/elastic.sce contains information
        of how to analyze the problem.
        
Note: So far, there is one major problem that students have 
faced with this software. The solution of the static 
displacement equation, K u = F, becomes weak for some mesh 
geometries, which are created by GMSH. If such a problem 
occurs, you may need to change your mesh geometry and/or
solution process (ex: LU decomposition, SVD, eigenvalue/eigenmode
, or other).

Good Luck.
