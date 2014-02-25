clean:
	find . -name "*.o" -o -name "*.out*" -o -name "*.so" | xargs rm -rf
commit  :
	git commit
	git push origin master
	git pull
save    :
	make clean
	cd .. && tar zcvf sandbox.tgz sandbox
revert  :
	git reset --hard HEAD
