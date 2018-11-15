gco_python: gco_src
	python3 setup.py build_ext -i
	cp pygco.cpython-36m-darwin.so ~/code/lab/practice-lab/Temporally_Coherent_Completion_of_Dynamic_Video/notebook/final/pygco/pygco.cpython-36m-darwin.so 

gco3d: gco_src
	python3 setup3d.py build_ext -i
	cp pygco3d.cpython-36m-darwin.so ~/code/lab/practice-lab/Temporally_Coherent_Completion_of_Dynamic_Video/notebook/final/pygco3d.cpython-36m-darwin.so 

test: test
	python3 setup3d.py build_ext -i
	python3 example.py