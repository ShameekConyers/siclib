#!/bin/zsh


case $1 in
	--l)
	rm -r build
	# rm -r pysiclib/**/*.pyi
	rm -r pysiclib/**/*.pyi
	mkdir build_tmp
	cd build_tmp &&
	cmake build .. &&
	cmake --build . &&
	stubgen -p _pysiclib -o . &&
	cp -r _pysiclib ../pysiclib &&
	cp _pysiclib.cpython-38-darwin.so ../pysiclib &&
	cd .. &&
	stubgen -p pysiclib -o . &&
	rm pysiclib/_pysiclib.cpython-38-darwin.so &&
	rm -r pysiclib/_pysiclib.pyi

	rm -r **/.mypy_cache
	;;
esac

/usr/local/anaconda3/bin/python setup.py bdist_wheel &&
/usr/local/anaconda3/bin/python -m pip install dist/pysiclib-0.0.6-cp38-none-any.whl --force-reinstall &&
unzip -l dist/*
