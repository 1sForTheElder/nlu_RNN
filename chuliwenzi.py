import sys
reload(sys)
sys.setdefaultencoding('utf8')
f = "xingfa.txt"
p = open(f)
for i in p.readline():
    print i