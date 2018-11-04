with open(r"Pagel_2010_FIN.txt") as f:
    data = f.readlines()

data_one_str = ''.join(data)

import re
results = re.findall('\([^)]*?[0-9]{4,4}[a-z]{0,1}\)',data_one_str,flags=re.MULTILINE & re.DOTALL)


re.findall('\([^)]*?[0-9]{4,4}[a-z]{0,1}\)',data_one_str,flags=re.MULTILINE & re.DOTALL)


numbers = re.finditer(r'[^0-9]{,30}([0-9]+[0-9.,]*).{,30}', data_one_str, re.MULTILINE & re.DOTALL)
for n in numbers:
    context = n.group(0)
    citations = re.findall('\(?[^)]*?[0-9]{4}[a-z]{0,1}\)?', context,re.MULTILINE & re.DOTALL)
    if not citations:
        position = "{} - {}:".format(n.start(1), n.end(1))
        print(position, n.group(1))
        
