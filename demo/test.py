# coding=utf-8
class person():
    def __init__(self,name,gender,birth,**kw):
        self.name=name
        self.gender=gender
        self.birth=birth
        for k, w in kw.iteritems():
            setattr(self,k,w)
        self.sayhi()

    def sayhi(self):
        print 'my name is',self.name
xiaoming = person('Xiao Ming', 'Male', '1991-1-1',job='student',tel='18089355',stdid='15010')
xiaohong = person('Xiao Hong', 'Female', '1992-2-2')
