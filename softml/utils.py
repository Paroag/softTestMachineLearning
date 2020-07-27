def nested_dictionnary(dic, nest_list, value, verbose = 0) :
	if verbose > 1 :
		print(f"parameters : {dic} {nest_list} {value}")
	if len(nest_list) == 1 :
		dic[nest_list[0]] = value
		if verbose > 1 :
			print(dic)
		return(dic)
	elif nest_list[0] not in dic :
		dic[nest_list[0]] = nested_dictionnary({}, nest_list[1:], value)
		if verbose > 1 :
			print(dic)
		return dic
	else : 
		dic[nest_list[0]] = nested_dictionnary(dic[nest_list[0]], nest_list[1:], value)
		if verbose > 1 :
			print(dic)
		return dic

if __name__ == "__main__" : 

    response = {}

    nested_dictionnary(response, ["StandardScaler","X"], 1, verbose = 2)
    nested_dictionnary(response, ["StandardScaler","Y"], 2, verbose = 2)
    nested_dictionnary(response, ["MinMaxScaler","X"], 3, verbose = 2)
    nested_dictionnary(response, ["MinMaxScaler","Y"], 4, verbose = 2)