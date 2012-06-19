cleanall:
	find . -name "*.o" -o -name "*.out*" | xargs rm -rf
commit  :
	git commit
	git push origin master
	git pull
save    :
	make cleanall
	cd .. && tar zcvf sandbox.tgz sandbox
revert  :
	hg reset HEAD
