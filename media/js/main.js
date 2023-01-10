// Теория - это когда все известно, но ничего не работает   
var icon_back = document.getElementsByClassName("icon_back");
icon_back[0].style.display = "none";

// USER IDENTIFICATION KEYSTROKE
var idk_block = document.createElement("div");
idk_block.id = "idl-block";
idk_block.style.display = "none";
idk_block.innerHTML = `<button type="button" 
                               id="see_posts" 
                               onclick="see_posts(this)" 
                               indicator="close">SEE ALL DATAS</button>
                       <div id="block_post"></div>
                       <br>
                       <div>
                           <input type="checkbox" id="test_user" name="test_user" onchange="getText(this)">
                           <label for="horns">test user</label>
                       </div>
                       <br>
                       <h3>Number of characters <span id="count_text">0</span>. Enter message:</h3>
                       <br>
                       <div id="show_value"></div>
                       <br> 
                       <div id="text_input" class="message_textarea" 
                                            role="textbox" 
                                            contenteditable="true" aria-multiline="true" aria-required="true" 
                                            style="background: white;font-size: 26px;margin: 9px auto;"></div>
                       <button type="button" onclick="send_test(this)" 
                                             indicator="send" 
                                             class="Button"
                                             style="margin: 4px auto; display: block;">SEND</button>`
                    
/*
`<button type="button" id="see_posts"  onclick="see_posts(this)">SEE ALL DATAS</button>
                       <button type="button" id="see_posts"  onclick="crate_data_all(this)">CRATE DATA ALL</button>
                       количество символов <span id="count_text">0</span>
                       <div id="block_post"></div>
                       <br>
                       <h3>Enter message:</h3>
                       <br>
                       <div id="show_value"></div>
                       <br> 
                       <div id="text_input" class="message_textarea" 
                                            role="textbox" 
                                            contenteditable="true" aria-multiline="true" aria-required="true" 
                                            style="background: white;font-size: 26px;margin: 9px auto;"></div>
                       <button type="button" onclick="send_test(this)" 
                                             indicator="send" 
                                             class="Button"
                                             style="margin: 4px auto; display: block;">SEND</button>`



You can use the indexOf method like this:
var index = array.indexOf(item);
if (index !== -1) {
  array.splice(index, 1);
}

Note: You'll need to shim it for IE8 and below
var array = [1,2,3,4]
var item = 3
var index = array.indexOf(item);
array.splice(index, 1);
console.log(array)

*/
                   
document.getElementById("testbox").appendChild(idk_block);

var text_input = document.getElementById("text_input");

var keyTimes = {};
var arr = [];
var idx_arr = 0;
text_input.onkeydown = text_input.onkeyup = text_input.onkeypress = handle;
function handle(e) {
    //console.log("WALL KEYPRESS", e, e.type, keyTimes);
    if (e.type == "keydown") {
        if (!keyTimes["key" + e.which]) {
            keyTimes["key" + e.which] = new Date().getTime();
        } else {
            if (e.key == "CapsLock" || e.key == "Alt" || e.key == "ArrowDown" || e.key == "ArrowUp" || e.key == "Control" || e.key == "Shift" || e.key=="ArrowLeft" || e.key=="ArrowRight") {
            
            } else if (e.key!="Backspace") {
                var _data = {"time_keydown": keyTimes["key" + e.which],
                                             "key_name":e.key, 
                                             "key_code":e.keyCode}           
                arr.push(_data);
                idx_arr++;
                
                var value_pure = '';
                var value_time = '';
                for (var i = 0; i < arr.length; i++) {
                    value_pure += arr[i].key_name;
                    value_time += arr[i].time_press;
                }
                show_value.innerHTML =`<div class="value_pure">${value_pure}</div>
                                      `; //<div class="value_time">${value_time}</div>
            }
        }
        if (e.key=="Backspace") {
            if (arr.length!=0 && idx_arr>0) {
                idx_arr--;
                arr.splice(idx_arr, 1); 
                count_text.innerHTML = arr.length;  
            }
            //-------------------------->
            var value_pure = '';
            for (var i = 0; i < arr.length; i++) {
                value_pure += arr[i].key_name;
            }
            show_value.innerHTML =`<div class="value_pure">${value_pure}</div>`            
        } else if (e.key=="ArrowLeft") {
            if (idx_arr > 0) {
                idx_arr--;
            }
        } else if (e.key=="ArrowRight") { 
            if (arr.length>idx_arr) {
                idx_arr++;
            }         
        } 
         
    } 
    if (e.type == "keyup") {
        if (keyTimes["key" + e.which]) {
            if (e.key=="ArrowLeft") {
            } else if (e.key=="ArrowRight") { 
            
            } else if (e.key=="Backspace" || e.key == "CapsLock" || e.key == "Alt") { 
            } else if (e.key == "ArrowDown" || e.key == "ArrowUp" || e.key == "Control" || e.key == "Shift") {
                                   
            } else {
                var time_up = new Date().getTime()
                var x = time_up - keyTimes["key" + e.which];
                //keyTimes["key" + e.which] = {"time_press":x / 1000.0, "key_name":e.key, "key_code":e.keyCode}
                var _data = {"time_keydown": keyTimes["key" + e.which] / 1000.0,
                             "time_press":x / 1000.0, 
                             "key_name":e.key, 
                             "key_code":e.keyCode, 
                             "time_keyup":time_up/1000.0}
                //arr.push(_data);
                arr.splice(idx_arr, 0, _data);
                delete keyTimes["key" + e.which];
//                console.log("KEYUP", e.key, x / 1000.0, idx_arr, arr.length);
                //------------->
                var value_pure = '';
                var value_time = '';
                for (var i = 0; i < arr.length; i++) {
                    value_pure += arr[i].key_name;
                    value_time += arr[i].time_press;
                }
                show_value.innerHTML =`<div class="value_pure">${value_pure}</div>
                                      `; //<div class="value_time">${value_time}</div>
                //------------>     
                idx_arr++;
                count_text.innerHTML = arr.length; 
            }
        } 
    }
}

function recording_key() {
    console.log('tick', arr)
    ws.send(JSON.stringify({'KEYPRESS': arr}));
    arr.length = 0 // удаляет все
    //arr = []; // удаляет все
    //keyTimes = {}; // удаляет все
}

function send_test(self) {
    console.log("SEND_TEST", document.getElementById("test_user").checked);
    var value_pure = '';
    for (var i = 0; i < arr.length; i++) {
        value_pure += arr[i].key_name;
    }    
    if (document.getElementById("test_user").checked) {
        ws.send(JSON.stringify({'event': 'send_test', 
                                'KEYPRESS': arr,
                                'text':value_pure,
                                'id_post': temp_id,
                                'test':document.getElementById("test_user").checked}));        
        
    } else {
        ws.send(JSON.stringify({'event': 'send_test', 
                                'KEYPRESS': arr,
                                'text':value_pure,
                                'test':document.getElementById("test_user").checked}));
    }
    text_input.innerText = "";
    arr.length = 0;
    show_value.innerHTML = "";
    keyTimes = {}
}
//----------------------------------->
var temp_id;
function getText(self) {
    if (self.checked) {
        console.log(self, self.checked);
        var http = createRequestObject();
        var linkfull = '/gettext/';
        if (http) {
            http.open('get', linkfull);
            http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
            http.onreadystatechange = function () {
                if (http.readyState == 4) {
                    var data = JSON.parse(http.responseText);
                    console.log(data);
                    block_post.innerHTML = data["text"];
                    temp_id = data["id_post"];
                }
            }
            http.send(null);  
        }        
    }
}


// тестирование гипотиез
function TESTKEY(self) {
    console.log(self)
}

// страница пользователя
function USER(self, id) {
    console.log(self, id)
    var http = createRequestObject();
    var linkfull = '/user_page/'+id;
    if (http) {
        http.open('get', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
                //var data = JSON.parse(http.responseText);
                //console.log(http.responseText);
                blockup.innerHTML = `<div id="node">
                                    <br>
                                    ${http.responseText}
                                    <br>
                                    <button onclick="close_div()">close</button>
                                </div>`
                //'<div id="node">' + http.responseText + '<a onclick="close_div()">закрыть</a></div>';
                blockup.style.display = "block";
                document.body.style.overflow = 'hidden';
                //idk_block.style.display = "none";
            }
        }
        http.send(null);  
    }
    
}

// выйти
function quit(){
    var cont = document.querySelector('body'); // ищем элемент с id
    var http = createRequestObject();
    if (http) {
        http.open('get', '/logout');
        http.onreadystatechange = function () {
            if(http.readyState == 4) {
                cont.innerHTML = http.responseText;
                isLoading = false;
                window.location.reload();
            }
        };
        http.send(null);
    } else {
        document.location = link;
    }
}

// Keystroke
function KEYSTROKE(self) {
    idk_block.style.display = "block";
    user_info.style.display = "none";
    //registration.style.display = "none";
    // кнопки
    document.getElementById("log_bt").style.display = "none";
    self.style.display = "none";
    // кнопка назад
    icon_back[0].style.display = "block"
    
    document.getElementsByClassName("testbox")[0].style.width = "740px";
    document.getElementById("logo24").style.display = "none";
    // отправлять каждую секунду данные
    //setInterval(recording_key, 1000);
};

// регистрация 
function REG(self) {
    //registration.style.display = "block";
    registration()
    self.style.display = "none";
    document.getElementById("log_bt").style.display = "none";
    // кнопка назад
    icon_back[0].style.display = "block"
};

function CHANGEUSER(self) {
    log_bt.style.display = "none";
    self.style.display = "none";
    try {user_info.style.display = "none";} catch (e) {};
    // кнопка назад    
    icon_back[0].style.display = "block"
    var crsv = getCookie('csrftoken'); // токен
    var http = createRequestObject();
    var linkfull = '/login/';
    if (http) {
        http.open('get', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.setRequestHeader('X-CSRFToken', crsv);
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
                main.innerHTML = http.responseText;
            }
        }
        http.send(null);  
    }    
};   

// кнопка назад     
icon_back[0].addEventListener('click', function(e) {
    // кнопка назад 
    location.reload();
});    

function registration() {
    var crsv = getCookie('csrftoken'); // токен
    var http = createRequestObject();
    var linkfull = '/registration/';
    if (http) {
        http.open('get', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.setRequestHeader('X-CSRFToken', crsv);
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
                document.getElementById("main").innerHTML = http.responseText;
            }
        }
        http.send(null);  
    } 
}

//---------------------------------->
// все данные
function see_posts(self) {
    console.log(self);
    if (self.getAttribute("indicator") == "close") {
        var crsv = getCookie('csrftoken'); // токен
        var http = createRequestObject();
        var linkfull = '/alldata/';
        if (http) {
            http.open('get', linkfull);
            http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
            http.setRequestHeader('X-CSRFToken', crsv);
            http.onreadystatechange = function () {
                if (http.readyState == 4) {
                    document.getElementById("block_post").innerHTML = http.responseText;
                    self.setAttribute("indicator", "open");
                    self.innerHTML = "CLOSE"
                    
                }
            }
            http.send(null);  
        } 
    } else {
        self.setAttribute("indicator", "close");
        document.getElementById("block_post").innerHTML = "";
        self.innerHTML = "SEE ALL DATAS";
    }
}

// подготовка всех данных
function crate_data_all(self) {
    var crsv = getCookie('csrftoken'); // токен
    var http = createRequestObject();
    var linkfull = '/cratealldata/';
    if (http) {
        http.open('get', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.setRequestHeader('X-CSRFToken', crsv);
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
                var data = JSON.parse(http.responseText);
                window.open(data["answer"], '_blank');
            }
        }
        http.send(null);  
    }

}

//---------------------------------->
const PORT = 8998;
var IP_ADDR = window.location.hostname;
var ws = new WebSocket("ws://"+ IP_ADDR +":"+PORT+"/wspage/");

//---------------------------------->

function createRequestObject() {
        try { return new XMLHttpRequest() }
        catch(e) {
            try { return new ActiveXObject('Msxml2.XMLHTTP') }
            catch(e) {
                try { return new ActiveXObject('Microsoft.XMLHTTP') }
                catch(e) { return null; }
            }
        }
}

function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie != '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function Register() {
    document.getElementById("FormReg").submit();
}

function close_div() {
    document.body.style.overflow = 'auto';
    //idk_block.style.display = "block";
    blockup.style.display = "none";
}

blockup.style.display = "none";
function see_data(self) {
    var id_p = self.getAttribute("post_id");
    var crsv = getCookie('csrftoken'); // токен
    var http = createRequestObject();
    var linkfull = '/data/'+ id_p;
    if (http) {
        http.open('get', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.setRequestHeader('X-CSRFToken', crsv);
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
                blockup.innerHTML = `<div id="node">
                                        <br>
                                        ${http.responseText}
                                        <br>
                                        <button onclick="close_div()">close</button>
                                    </div>`
                //'<div id="node">' + http.responseText + '<a onclick="close_div()">закрыть</a></div>';
                blockup.style.display = "block";
                document.body.style.overflow = 'hidden';
                //idk_block.style.display = "none";
            }
        }
        http.send(null);  
    }     
}
                
//ws.binaryType = 'arraybuffer';
ws.onopen = function() {
    console.log("connection was established");
};

ws.onmessage = function(data) {
    var message_data = JSON.parse(data.data);
    if (message_data["status"] == "send_test") {
        console.log(message_data, block_post);
//        block_post.innerHTML += `<button type="button" onclick="see_data(this)" 
//                                         indicator="send" class="Button" 
//                                         style="margin: 4px auto; display: block;" post_id="${message_data["post_id"]}">
//                                         Посмотреть статисику поста #${message_data["post_id"]}, 
//                                         пользователя ${message_data["user_post"]}</button>`

        block_post.innerHTML += `<button type="button" onclick="see_data(this)" 
                                         indicator="send" class="Button" 
                                         post_id="${message_data["post_id"]}">
                                         Посмотреть статисику поста #${message_data["post_id"]}, 
                                         пользователя ${message_data["user_post"]}</button>`
    } else if (message_data["status"] == "send_test_p") {
        blockup.innerHTML = `<div id="node">
                                        <br>
                                        ${message_data["html"]}
                                        <br>
                                        <button onclick="close_div()">close</button>
                                    </div>`        
        blockup.style.display = "block";
        document.body.style.overflow = 'hidden';
    } else if (message_data["status"] == "MoreData") {
    } else if (message_data["status"]=="Done") { 
    }
        
};

