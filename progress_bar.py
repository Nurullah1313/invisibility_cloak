from tqdm import tqdm

for i in tqdm(range(int(9e6))):
    pass


for i in tqdm(range(int(9e6)), ascii = True, desc = 'hello'):
    pass


# terminale tqdm -help yazarak daha fazla bilgiye erişebilirsin

