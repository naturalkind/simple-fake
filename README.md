# simple-fake-detect

install   

Redis 6   
```
sudo add-apt-repository ppa:redislabs/redis
sudo apt-get update
sudo apt-get install redis

/etc/init.d/redis-server restart
```

Виртуальная среда для работы с Django 3
```
python3 -m venv <myenvname>

source <myenvname>/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

./manage.py makemigrations

./manage.py migrate auth

./manage.py migrate --run-syncdb

./manage.py createsuperuser

./manage.py dumpdata > data_dump.json
```

run   

```
python manage.py runserver IP:PORT

```

test   
```
./manage.py shell < stat_utils/check_data1.py
```

### Нужно сделать:
- после регистрации, сразу предложить напечатать проверочный тест :heavy_check_mark:   
- при авторизации напечатать любой текст и оценить по алгоритмам - кто это   
- пока алгоритм производит вычисления, визуально отображать :heavy_check_mark:   
- добавить поддержку ввода с мобильных устройств   
- сформировать текст на основе данных - пары.csv :heavy_check_mark:   
- подключить v4, проверить   
- p-value 100, отсеивать   
- при регистрации нет отображения имени юзера   
- исправление ошибок   
- сохранение текстов авторизации   
- выводить текст из базы, с возможностью переключения :heavy_check_mark:   
