# Legacy Scilab / MATLAB Sources

This directory preserves the original Scilab FemLab wrapper prepared by **G. Turan** at IYTE,
derived from the MATLAB FemLab teaching toolbox by O. Hededal & S. Krenk (Aalborg University).

These files are kept for **historical reference and parity checks** against the Python port in `src/femlab/`.

## Structure

```
legacy/
├── scilab/
│   ├── builder.sce          # Scilab toolbox builder script
│   ├── builder_old.sce      # Previous builder version
│   ├── loader.sce           # Scilab toolbox loader
│   ├── macros/              # .sci function libraries (FEM elements, solvers, plotting)
│   ├── examples/            # .sce example scripts (cantilever, elastic, lagrange mult.)
│   └── help/                # .htm help pages for each function
├── mesh/                    # Gmsh geometry and mesh files (.geo, .msh) + intro docs
└── doc/
    ├── README_first.txt     # Original project readme
    └── manual.ps            # PostScript version of the manual (PDF in docs/)
```

## License Note

The legacy source files contain copyright notices but no obvious standalone license file in the imported material.
Until upstream licensing is clarified, treat the legacy MATLAB/Scilab content as historical reference material.
