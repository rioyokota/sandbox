clean:
	find . -name "*.o" -o -name "*.out*" | xargs rm -rf
cleandat:
	find . -name "*.dat" -o -name "*.dot" -o -name "*.svg" | xargs rm -rf
cleanlib:
	find . -name "*.a" -o -name "*.so" | xargs rm -rf
cleanall:
	make clean
	make cleandat
	make cleanlib
commit  :
	git commit
	git push origin master
	git pull
save    :
	make cleanall
	cd .. && tar zcvf sandbox.tgz sandbox
revert  :
	hg reset --hard HEAD
