

def fn1(**kwargs):

    print(kwargs)
    fn2(**kwargs)

def fn2(**kwargs):

    b = kwargs.get('b',20)
    print(b)


fn1(a=8,b=9)



exit(0)


#
