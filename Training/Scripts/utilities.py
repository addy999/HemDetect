def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

def getUniqueBtwStrings(strings):
    
    common_factor = longestSubstringFinder(strings[0], strings[1])
    unique = []
    for string in strings:
        r = string.replace(common_factor, "").replace("_epoch", "").replace(".pt","")
        unique.append(int(r))
    return unique

def findBs(string):
    end_idx = string.find("_bs")
    return int(string[:end_idx])