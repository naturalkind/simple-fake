import time
import numpy as np

alf_ = ord('а')
abc_ = ''.join([chr(i) for i in range(alf_, alf_+32)])
abc_ += " ,."

#abc_id = [i for i in range(len(abc_))]

combination = ["ст", "то", "но", "на", "по", "ен", "ни", "не", "ко", "ра", "ов", "ро", "го", "ал",
               "пр", "ли", "ре", "ос", "во", "ка", "ер", "от", "ол", "ор", "та", "ва", "ел", "ть",
               "ет", "ом", "те", "ло", "од", "ла", "ан", "ле", "ве", "де", "ри", "ес", "ат", "ог",
               "ль", "он", "ны", "за", "ти", "ит", "ск", "ил", "да", "ой", "ем", "ак", "ме", "ас",
               "ин", "об", "до", "че", "мо", "ся", "ки", "ми", "се", "тр", "же", "ам", "со", "аз",
               "нн", "ед", "ис", "ав", "им", "ви", "тв", "ар", "бы", "ма", "ие", "ру", "ег", "бо",
               "сл", "из", "ди", "чт", "вы", "вс", "ей", "ия", "пе", "ик", "ив", "сь", "ое", "их",
               "ча", "ну", "мы"] # 101   

control_text = """повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим,
как будет изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в
которых работает оператор. желательно, чтобы сначала работал в одних и тех же условиях.
повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим, как будет
изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в которых
работает оператор. желательно, чтобы сначала работал в одних и тех же условиях.
повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим, как будет
изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в которых
работает оператор. желательно, чтобы сначала работал в одних и тех же условиях."""


combination_idx = np.zeros((len(combination), 2))
for ix, i in enumerate(combination):
    #print (abc_.index(i[0]), abc_.index(i[1]))
    combination_idx[ix, :] = [abc_.index(i[0]), abc_.index(i[1])]

#print (combination_idx)


def find_all_loc(vars, key):
    pos = []
    start = 0
    end = len(vars)
    while True: 
        loc = vars.find(key, start, end)
        if  loc is -1:
            break
        else:
            pos.append(loc)
            start = loc + len(key)
    return pos


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

  
def func_np(l):
    bss = 0
    lsd = []
    while bss <= l:  
        print (bss, h[1])
#        if control_text[b+1] == h[1]:
#            lsd.append(b)
        bss+=1
    return lsd
        
start_for = time.time()
# numpy вариант
np_zeros = np.zeros((len(control_text), 1)) #len(abc_)
for ig, g in enumerate(control_text):
    if g != "\n":
        np_zeros[ig, 0] =  abc_.index(g) 

for ih, h in enumerate(combination):
    if control_text.count(h) != 0:
        start_t0 = time.time()
        idx = find_all_loc(control_text, h) # РАБОТАЕТ БЫСТРЕЕ !!!
        end_time0 = time.time()
        print (f"FIND_ALL_LOC -> {h} <-", round((end_time0-start_t0) * 1000, 5), "------------------", idx)
        
        start_t0 = time.time()
        idx = indices(control_text, h)
        end_time0 = time.time()
        print (f"INDICES -> {h} <-", round((end_time0-start_t0) * 1000, 5), "------------------", idx)
        
        start_t1 = time.time()
#        # numpy вариант
#        np_zeros = np.zeros((len(control_text), 1)) #len(abc_)
#        for ig, g in enumerate(control_text):
#            if g != "\n":
#                np_zeros[ig, 0] =  abc_.index(g) 

               
        ixs1 = np.where(np_zeros==combination_idx[ih, 0])[0]#.tolist()
        lsd = [] #func_np(ixs1.shape[0])
        for b in ixs1:
            if control_text[b+1] == h[1]:
                #print (control_text[b+1], h[1])
                lsd.append(b)
        end_time1 = time.time()
        #print (np_zeros.shape)
        print (f"Мой вариант NUMPY -> {h} <-", round((end_time1-start_t1) * 1000, 5), "------------------", lsd)
        
end_time_final = time.time()
print (end_time_final-start_for)    
#https://translated.turbopages.org/proxy_u/en-ru.ru.3b9232e5-63a9d041-1c25eb53-74722d776562/https/stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list-    
    
#https://translated.turbopages.org/proxy_u/en-ru.ru.58741e9b-63a988cd-c065005c-74722d776562/https/stackoverflow.com/questions/991350/counting-repeated-characters-in-a-string-in-python    
  
 
