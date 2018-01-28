c__l = 0.1;

Point(1) = {0.0, 0.0, 0, c__l};
Point(2) = {1.0, 0.0, 0, c__l};
Point(3) = {1.5, 0.866, 0, c__l};
Point(4) = {0.5, 0.866, 0, c__l};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(5) = {3, 4, 1, 2};  // perimeter

Plane Surface(6) = {5};

Physical Line("Dirichlet_1") = {4};
Physical Line("Dirichlet_2") = {2};
Physical Line("Dirichlet_3") = {3};
Physical Line("Neumann_4") = {1};
Physical Surface("Fluid") = {6};
