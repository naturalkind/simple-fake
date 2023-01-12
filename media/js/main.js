// Теория - это когда все известно, но ничего не работает   
var icon_back = document.getElementsByClassName("icon_back");
icon_back[0].style.display = "none";

// отображать выполнение
var t_el = document.createElement("div");
t_el.id = "loader";
t_el.style.display = "block";
t_el.style.margin = "0 auto";

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
                       <p>Elapsed time: <span id="time">0</span>s</p>
                       <p>Number of characters: <span id="count_text">0</span></p>
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
                    
                   
document.getElementById("testbox").appendChild(idk_block);

var text_input = document.getElementById("text_input");

var arr = [];
//"Backspace", "ArrowLeft", "ArrowRight", 
list_exept = ["ArrowDown", "ArrowUp", "CapsLock", "Alt", "Control", "Shift"]
// подготовка текста к отправке на сервер
var idx_arr = 0;
text_input.onkeydown = text_input.onkeyup = text_input.onkeypress = text_input.onclick = handle;
//text_input.textContent

// позиция буквы в тексте div role=textbox https://stackoverflow.com/questions/8105824/determine-the-position-index-of-a-character-within-an-html-element-when-clicked
function getSelectionPosition () {
  var selection = window.getSelection();
  idx_arr = selection.focusOffset
}

//--------------------------->


var start_all_time;
var end_all_time;
var focusHandler = function() {
    console.log("Focus");
    timer.start();
    start_all_time = new Date().getTime()/1000.0;
}
var blurHandler = function() {
    console.log("Focus End");
    end_all_time = new Date().getTime()/1000.0;
    timer.stop();
    timer.reset();
}

text_input.onfocus = focusHandler;
text_input.onblur = blurHandler;

// таймер https://stackoverflow.com/questions/29971898/how-to-create-an-accurate-timer-in-javascript
class Timer {
  constructor () {
    this.isRunning = false;
    this.startTime = 0;
    this.overallTime = 0;
  }

  _getTimeElapsedSinceLastStart () {
    if (!this.startTime) {
      return 0;
    }
  
    return Date.now() - this.startTime;
  }

  start () {
    if (this.isRunning) {
      return console.error('Timer is already running');
    }

    this.isRunning = true;

    this.startTime = Date.now();
  }

  stop () {
    if (!this.isRunning) {
      return console.error('Timer is already stopped');
    }

    this.isRunning = false;

    this.overallTime = this.overallTime + this._getTimeElapsedSinceLastStart();
  }

  reset () {
    this.overallTime = 0;

    if (this.isRunning) {
      this.startTime = Date.now();
      return;
    }

    this.startTime = 0;
  }

  getTime () {
    if (!this.startTime) {
      return 0;
    }

    if (this.isRunning) {
      return this.overallTime + this._getTimeElapsedSinceLastStart();
    }

    return this.overallTime;
  }
}

const timer = new Timer();

setInterval(() => {
  const timeInSeconds = Math.round(timer.getTime() / 1000);
  document.getElementById('time').innerText = timeInSeconds;
}, 100)

//--------------------------->
var tKey = {}
function handle(e) {
    if (e.type == "click") {
        getSelectionPosition ();
//        console.log("WALL KEYPRESS", e, e.type, e.anchorOffset);
    }
    if (list_exept.indexOf(e.key) == -1) {
        if (e.type == "keydown") {
            if (e.key=="Backspace") {
                if (arr.length!=0 && idx_arr>0 ) {
                    idx_arr--;
                    arr.splice(idx_arr, 1);
                    count_text.innerHTML = arr.length;  
                    let value_pure = '';
                    for (var i = 0; i < arr.length; i++) {
                        value_pure += arr[i].key_name;
                    }
                    show_value.innerHTML = value_pure;
                }
                 
                //`<div class="value_pure">${text_input.innerText}</div>`;
            } else if (e.key=="ArrowLeft") {
                if (idx_arr > 0) {
                    idx_arr--;
                }
            } else if (e.key=="ArrowRight") { 
                if (arr.length>idx_arr) {
                    idx_arr++;
                }         
            } else {
                //-------------------------------->
                var keyTimes = {};
                keyTimes["key_code"] = e.keyCode;
                let K;
                if (e.key =="Enter") { K = " " } else { K = e.key };
                keyTimes["key_name"] = K;
                keyTimes["time_keydown"] = new Date().getTime()/1000.0;
                //arr.push(keyTimes); 
                arr.splice(idx_arr, 0, keyTimes);
                if (!tKey[e.key]) {
                    tKey[e.key] = [idx_arr];
                } else {
                    tKey[e.key].push(idx_arr);
                }
                //show_value.innerHTML = text_input.innerHTML   
                show_value.innerHTML += K;
                count_text.innerHTML = arr.length;                   
                idx_arr++  

//                console.log(e.key, list_exept.indexOf(e.key), keyTimes, arr.length, text_input.innerText.length);
            }
        }
        if (e.type == "keyup") {
            if (arr.length>0) {
                let time_up = new Date().getTime()/1000.0;
                try {
                    for (var i = 0; i < tKey[e.key].length; i++) {
                        let rev_idx = (tKey[e.key].length-1)-i;
                        arr[tKey[e.key][i]]["time_keyup"] = time_up;
                        arr[tKey[e.key][i]]["time_press"] = time_up - arr[tKey[e.key][rev_idx]]["time_keydown"];
                    }
                    delete tKey[e.key];
                } catch (e) {}
            }
        }
//        if (arr.length == text_input.innerText.length) {
//            var value_pure = "";
//            for (var i = 0; i < arr.length; i++) {
//                value_pure += arr[i].key_name;
//            }
//            
//            show_value.innerHTML = text_input.innerHTML;
////            show_value.innerHTML =`<div class="value_pure">${value_pure}</div>`;
////            show_value.innerHTML =`<div class="value_pure">${text_input.innerText}</div>`;
//            count_text.innerHTML = arr.length; 
//            console.log(text_input.innerText.replace(/\s+/g, ' ').trim(), "<---->", value_pure,
//                        text_input.innerText.replace(/\s+/g, ' ').trim() === value_pure.trim());
//        }
    }
}

function recording_key() {
    console.log('tick', arr)
    ws.send(JSON.stringify({'KEYPRESS': arr}));
    arr.length = 0 // удаляет все
    //arr = []; // удаляет все
    //keyTimes = {}; // удаляет все
}

var checked_ = false;

function send_test(self) {
    try {
        checked_  = document.getElementById("test_user").checked;
    } catch(e) {}
    ws.send(JSON.stringify({'event': 'send_test', 
                            'KEYPRESS': arr,
                            'text':text_input.innerText,
                            'id_post': temp_id,
                            'test':checked_}));              
    
    text_input.innerText = "";
    arr.length = 0;
    count_text.innerHTML = 0;
    show_value.innerHTML = "";
    show_value.appendChild(t_el);
    keyTimes = {}            
}


//function send_test(self) {
//    if (arr.length == text_input.innerText.length) {
//        var value_pure = "";
//        for (var i = 0; i < arr.length; i++) {
//            value_pure += arr[i].key_name;
//        }
//        if (text_input.innerText.trim() === value_pure.trim()) {
//            console.log(text_input.innerText, "<---->", value_pure, text_input.innerText.trim() === value_pure.trim(), end_all_time-start_all_time, checked_);
//            try {
//                checked_  = document.getElementById("test_user").checked;
//            } catch(e) {}
//            ws.send(JSON.stringify({'event': 'send_test', 
//                                    'KEYPRESS': arr,
//                                    'text':text_input.innerText,
//                                    'id_post': temp_id,
//                                    'test':checked_}));              
//            
//            text_input.innerText = "";
//            arr.length = 0;
//            count_text.innerHTML = 0;
//            show_value.innerHTML = "";
//            show_value.appendChild(t_el);
//            keyTimes = {}            
//            
//            
//        }
//    }
//}


//----------------------------------->
var temp_id;
function getText(self) {
    if (self.checked) {
        //console.log(self, self.checked);
        var http = createRequestObject();
        var crsv = getCookie('csrftoken'); // токен
        var linkfull = '/gettext/';
        if (http) {
            http.open('post', linkfull);
            http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
            http.setRequestHeader('X-CSRFToken', crsv);
            http.onreadystatechange = function () {
                if (http.readyState == 4) {
                    var data = JSON.parse(http.responseText);
                    //console.log(data);
                    block_post.innerHTML = data["text"];
                    temp_id = data["id_post"];
                }
            }
            let data = JSON.stringify({"post_id": self.getAttribute("post_id")});
            http.send(data);  
        }        
    } else {
        console.log("UNCHECKED")
        block_post.innerHTML = "";
        document.getElementById("see_posts").setAttribute("indicator", "close");
        document.getElementById("see_posts").innerHTML = "SEE ALL DATAS";
    }
}


// тестирование гипотиез
function TESTKEY(self) {
    console.log(self)
}

// страница пользователя
function USER(self, id) {
    var http = createRequestObject();
    var linkfull = '/user_page/'+id;
    if (http) {
        http.open('get', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
                //var data = JSON.parse(http.responseText);
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
    registration();
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
        checked_ = false;
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
                                        <input type="checkbox" id="test_user" name="test_user" onchange="getText(this)" post_id="${id_p}">
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
    show_value.removeChild(t_el);
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
    } else if (message_data["status"]=="Done") { 
        
    }
        
};

