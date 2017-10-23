#!/bin/bash

cd src
make clean
make
cd ..
bin/camus zz_test.inp_membrane
