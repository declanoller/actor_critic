from datetime import datetime
from os import mkdir
from copy import copy,deepcopy
import time

def getDateString():
    return(datetime.now().strftime("%H-%M-%S"))


def makeDir(dir_name):
    #Even if this is in a library dir, it should make the dir
    #in the script that called it.
    mkdir(dir_name)
    return(dir_name)


def makeDateDir():
    #Just creates a dir with the current date for its name
    ds = getDateString()
    makeDir(ds)
    return(ds)


def makeLabelDateDir(label):
    #You give it a label, and it creates the dir label_datestring
    dir = label + '_' + getDateString()
    makeDir(dir)
    return(dir)


def combineDirAndFile(dir,file):

    if dir[-1] == '/':
        return(dir + file)
    else:
        return(dir + '/' + file)


def paramDictToFnameStr(param_dict):
    #Creates a string that can be used as an fname, separated by
    #underscores. If a param has the value None, it isn't included.
    pd_copy = copy(param_dict)
    for k,v in pd_copy.items():
        if type(v).__name__ == 'float':
            pd_copy[k] = '{:.2f}'.format(v)

    params = [str(k)+'='+str(v) for k,v in pd_copy.items() if v is not None]
    return('_'.join(params))

def paramDictToLabelStr(param_dict):
    #Creates a string that can be used as an fname, separated by
    #', '. If a param has the value None, it isn't included.
    pd_copy = copy(param_dict)
    for k,v in pd_copy.items():
        if type(v).__name__ == 'float':
            pd_copy[k] = '{:.2f}'.format(v)

    params = [str(k)+'='+str(v) for k,v in pd_copy.items() if v is not None]
    return(', '.join(params))


def listToFname(list):
    return('_'.join(list))


def parseSingleAndListParams(param_dict,exclude_list):

    #This is useful for if you want to do multiple runs, varying one or
    #several parameters at once. exclude_list are ones you don't want to
    #include in the parameters in the tuple.

    #It returns a list of the parameters that are varied,
    #and a list of dictionaries that can be directly passed to a function.
    list_params = []
    single_params = {}
    ziplist = []

    for k,v in param_dict.items():
        if type(v).__name__ == 'list':
            list_params.append(k)
            ziplist.append(v)
        else:
            if k not in exclude_list:
                single_params[k] = v

    param_tups = list(zip(*ziplist))

    vary_param_dicts = []
    vary_param_tups = []
    for tup in param_tups:
        temp_dict = dict(zip(list_params,tup))
        temp_kw = {**single_params,**temp_dict}
        vary_param_tups.append(temp_dict)
        vary_param_dicts.append(temp_kw)

    return(list_params,vary_param_dicts,vary_param_tups)



def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def getCurTimeObj():
    return(datetime.now())


def getTimeDiffStr(start_time):
    #Gets the time diff in a nice format from the start_time.
    diff = datetime.now() - start_time

    return(strfdelta(diff,'{hours} hrs, {minutes} mins, {seconds} s'))








#
