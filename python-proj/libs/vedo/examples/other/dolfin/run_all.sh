#!/bin/bash
# source run_all.sh
#
##########################
echo Running ascalarbar.py
python3 ascalarbar.py

echo Running collisions.py
python3 collisions.py

echo Running calc_surface_area.py
python3 calc_surface_area.py

echo Running markmesh.py
python3 markmesh.py

echo Running demo_submesh.py
python3 demo_submesh.py

echo Running elastodynamics.py
python3 elastodynamics.py

echo Running elasticbeam.py
python3 elasticbeam.py

echo Running pointLoad.py
python3 pointLoad.py

echo Running nodal_u.py
python3 nodal_u.py

echo Running read_image.py
python3 read_image.py


######################################

echo Running ex03_poisson.py
python3 ex03_poisson.py

echo Running ex04_mixed-poisson.py
python3 ex04_mixed-poisson.py

echo Running ex06_elasticity1.py
python3 ex06_elasticity1.py

echo Running ex06_elasticity2.py
python3 ex06_elasticity2.py

echo Running ex07_stokes-iterative.py
python3 ex07_stokes-iterative.py


######################################

echo Running ft04_heat_gaussian.py
python3 ft04_heat_gaussian.py

echo Running navier-stokes_lshape.py
python3 navier-stokes_lshape.py

echo Running ft09_reaction_system.py
python3 ft09_reaction_system.py

echo Running demo_cahn-hilliard.py
python3 demo_cahn-hilliard.py

echo Running heatconv.py
python3 heatconv.py

echo Running awefem.py
python3 awefem.py

echo Running demo_eigenvalue.py
python3 demo_eigenvalue.py

