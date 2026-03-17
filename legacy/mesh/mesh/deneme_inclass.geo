// Gmsh project created on Mon Apr  7 13:54:24 2014
Point(1) = {-0.6, -0.4, 0, 1.0};
Point(2) = {0.4, -0.7, 0, 1.0};
Point(3) = {0.9, -0.1, 0, 1.0};
Point(4) = {0.5, 0.6, 0, 1.0};
Point(5) = {-0.7, 0.5, 0, 1.0};
Point(6) = {-1.1, 0.1, 0, 0.02};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {4, 3};
BSpline(4) = {4, 5, 6};
Line(5) = {1, 6};
Line(6) = {4, 1};
Line Loop(7) = {1, 2, -3, 6};
Plane Surface(8) = {7};
Line Loop(9) = {4, -5, -6};
Plane Surface(10) = {9};
