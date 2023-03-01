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
                               indicator="close"
                               style="display:none;">SEE ALL DATAS</button>
                       <div id="block_post"></div>
                       <br>
                       <p>Elapsed time: <span id="time">0</span>s</p>
                       <p>Number of characters: <span id="count_text">0</span>
                       <span id="count_text_our" style="display:none;">0</span></p>
                       <p style="display:none;">Percentage of filling: <span id="count_text_per">0</span>%</p> 
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

var arr_bad = [];
//"Backspace", "ArrowLeft", "ArrowRight", 
list_exept = ["CapsLock", "Alt", "Control", "Shift", "Insert"]
// подготовка текста к отправке на сервер
var idx_arr = 0;
text_input.onkeydown = text_input.onkeyup = text_input.onkeypress = text_input.onclick = handle;
//text_input.textContent

// позиция буквы в тексте div role=textbox 
// https://stackoverflow.com/questions/8105824/determine-the-position-index-of-a-character-within-an-html-element-when-clicked

// WORK
//function getSelectionPosition () {
//  var selection = window.getSelection();
//  idx_arr = selection.focusOffset
//}


// WORK2
function getCaretCharOffset(element) {
  var caretOffset = 0;
  if (window.getSelection) {
    var range = window.getSelection().getRangeAt(0);
    var preCaretRange = range.cloneRange();
    preCaretRange.selectNodeContents(element);
    preCaretRange.setEnd(range.endContainer, range.endOffset);
    caretOffset = preCaretRange.toString().length;
  } 
  return caretOffset;
}



//--------------------------->


var start_all_time;
var end_all_time;
var focusHandler = function() {
//    console.log("Focus");
    timer.start();
    start_all_time = new Date().getTime()/1000.0;
}
var blurHandler = function() {
//    console.log("Focus End");
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
//https://stackoverflow.com/questions/3972014/get-contenteditable-caret-position

//https://translated.turbopages.org/proxy_u/en-ru.ru.f36662aa-63cc1d4d-b749bddc-74722d776562/https/stackoverflow.com/questions/1444477/keycode-and-charcode

//https://github.com/iliakan/javascript-tutorial-ru/blob/master/2-ui/3-event-details/9-keyboard-events/article.md

//https://stackoverflow.com/questions/10282314/what-is-a-r-and-why-would-one-use-it-with-a-n

//https://stackoverflow.com/questions/3427132/how-to-get-first-character-of-string

//https://ru.stackoverflow.com/questions/504475/Подсвечивание-текста-при-вводе

var tKey = {}

function getCaret() {
  const editable = document.getElementById('text_input');
  // collapse selection to end
  window.getSelection().collapseToEnd();

  const sel = window.getSelection();
  const range = sel.getRangeAt(0);

  // get anchor node if startContainer parent is editable
  let selectedNode = editable === range.startContainer.parentNode
    ? sel.anchorNode 
    : range.startContainer.parentNode;

  if (!selectedNode) {
    console.log(`caret: 0, line: 0`);
    return;
  }

  // select to top of editable
  if (editable.firstChild) {
    range.setStart(editable.firstChild, 0);
  }

  // do not use 'this' sel anymore since the selection has changed
  const content = window.getSelection().toString();
  const text = JSON.stringify(content);
  const lines = (text.match(/\\n/g) || []).length;

  // clear selection
  window.getSelection().collapseToEnd();
  idx_line = lines;
}
//------------------------------------------>

let bp = document.getElementById("block_post");
let textOriginal = bp.textContent;

let span = document.createElement('span');
span.classList.add('highlight');
let hasChanges = false; 


function repl (match) {
    span.innerHTML = match;
    return span.outerHTML;
}

var temp_text_line;
var P_string = 0;
//------------------------------------------>
var idx_line = 0;
var searchKey = "";


function handle(e) {
    document.getElementById("show_value").innerHTML = text_input.innerHTML;
    document.getElementById("count_text").innerHTML = arr.length;
    document.getElementById("count_text_our").innerText = textOriginal.length;
//    getSelectionPosition(); //text_input.textContent
    getCaret();
    idx_arr = getCaretCharOffset(e.target)+idx_line;
    var charCode = e.which || e.keyCode;
    let textNew = textOriginal;
    
//    console.log("Start--->", idx_arr);
    if (list_exept.indexOf(e.key) == -1) {
        if (e.type == "keydown") {
            if (e.key=="Backspace") {
//              arr.length!=0 &&     
                if (arr.length!=0 && idx_arr!=0) {
//                if (text_input.innerText.length >=0) {
                    idx_arr--;
                    // WORK
//                    console.log("Backspace", idx_arr, arr[idx_arr], textOriginal[idx_arr-1]);
//                    if (textOriginal[idx_arr]===arr[idx_arr].key_name) {
//                        if (idx_arr==0){
//                            bp.innerHTML = textNew.replace(textOriginal[idx_arr-1], arr[idx_arr].key_name);
//                        } else {
//                            bp.innerHTML = textOriginal.slice(0, idx_arr-1) +textNew.substr(idx_arr-1).replace(textOriginal[idx_arr-1], repl);
//                            //bp.innerHTML = textNew.replace(textOriginal[idx_arr-1], repl);
//                        }
//                        
//                    }

                    // WORK
                    if (textOriginal[idx_arr]===arr[idx_arr].key_name) {
                        searchKey = textOriginal.slice(0, idx_arr);
                        temp_text_line = repl(textOriginal.slice(0, idx_arr));
                        console.log(temp_text_line);
                        bp.innerHTML = temp_text_line+textNew.substr(idx_arr);
                        
                    }

                    //
                    arr.splice(idx_arr, 1);
                    document.getElementById("count_text").innerHTML = arr.length;  

                }
            } else if (e.key == "ArrowRight") {   
            } else if (e.key == "ArrowLeft") {  
            } else if (e.key == "ArrowUp") {  
            } else if (e.key == "ArrowDown") {   
            } else {
                //-------------------------------->
                var keyTimes = {};
                //keyTimes["key_code"] = e.keyCode;
                let K;
                if (e.key =="Enter") { 
                    //K = "\n"; 
//                    var charCode = e.which || e.keyCode;
                    K = String.fromCharCode(charCode); //String.fromCodePoint
//                    console.log(".....", K);
                } else { K = e.key };
                keyTimes["key_name"] = K;
                keyTimes["time_keydown"] = new Date().getTime()/1000.0;
                keyTimes["time_keyup"] = new Date().getTime()/1000.0;
                arr.splice(idx_arr, 0, keyTimes);
//                arr.push(keyTimes);
                if (!tKey[e.key]) {
                    tKey[e.key] = [idx_arr];
                } else {
                    tKey[e.key].push(idx_arr);
                }
            }
//            console.log("KEYDOWN", e.key, arr.length, text_input.innerText.length, idx_arr);
        }
        if (e.type == "keypress") {
            if (textOriginal[idx_arr]===String.fromCharCode(charCode)) {
                searchKey += String.fromCharCode(charCode);
                if (textOriginal.slice(0, idx_arr+1)===searchKey) {
                    console.log(textOriginal.slice(0, idx_arr+1), searchKey);
                    temp_text_line = repl(textOriginal.slice(0, idx_arr+1));
                    bp.innerHTML = temp_text_line+textNew.substr(idx_arr+1);

                    P_string = (textOriginal.slice(0, idx_arr+1).length*100)/textNew.length;
                    document.getElementById("count_text_per").innerText = P_string.toFixed(1);
                }

                //// WORK
//                console.log(textOriginal.slice(0, idx_arr));
//                bp.innerHTML = textOriginal.slice(0, idx_arr) + textNew.substr(idx_arr).replace(textOriginal[idx_arr], repl);

                //// WOKK
//                let searchKey = text_input.textContent;
//                let regExp = new RegExp(searchKey == '\\' ? '' : searchKey, 'gi');
//                let textNew = textOriginal;
//                // Если нет совпадений в тексте
//                if (!regExp.test(textOriginal)) {
//                 if (hasChanges) {
//                   hasChanges = false;
//                     bp.innerHTML = textNew;
//                 }
//                 return true;
//                }
//                hasChanges = true;
//                function repl (match) {
//                 span.innerHTML = match;
//                 return span.outerHTML;
//                }
//                if(text_input.innerText.trim() === "" ) {
//                  bp.innerHTML = textNew;
//                }
//                else {
//                  bp.innerHTML = textNew.replace(regExp, repl);
//                }


            } 
//            document.getElementById("show_value").innerHTML = text_input.innerHTML;
//            console.log("KEYPRESS", e.key, arr.length, text_input.innerText.length);
        }
        if (e.type == "keyup") {
            if (arr.length>0) {
                let time_up = new Date().getTime()/1000.0;
                if (tKey[e.key]) {
                    for (var i = 0; i < tKey[e.key].length; i++) {
                        let rev_idx = (tKey[e.key].length-1)-i;
                        arr[tKey[e.key][i]]["time_keyup"] = time_up;
                        arr[tKey[e.key][i]]["time_press"] = time_up - arr[tKey[e.key][rev_idx]]["time_keydown"];
                    }
                    delete tKey[e.key];
                }
            }
//            console.log("KEYUP", e.key, arr.length, idx_arr, idx_line);
        }
    } else {
        if (e.type == "keydown") {
            arr_bad.push([e.key, idx_arr]);
        }
//        console.log(e.key, idx_arr);
    }
}



//function send_for_log(self) {
//    let value_pure = '';
//    for (var i = 0; i < arr.length; i++) {
//        value_pure += arr[i].key_name;
//    }
//    
//    console.log(arr, text_input.innerText, value_pure, idx_arr, arr_bad)
//    console.log(text_input.innerText.replace(/\s+/g, ' ').trim(), "<---->", value_pure.replace(/\s+/g, ' ').trim(),
//                text_input.innerText.replace(/\s+/g, ' ').trim() === value_pure.replace(/\s+/g, ' ').trim());
//}

//--------------------------------->
var result_ = "start"
function recording_key() {
    //console.log('tick', arr)
    let data = JSON.stringify({'event':'KEYPRESS',
                               'KEYPRESS': arr,
                               'KEYPRESS_BAD': arr_bad,
                               'text':text_input.innerText});   
    if (result_ == "start") {
        ws.send(data);
    }
    
    //arr.length = 0 // удаляет все
    //arr = []; // удаляет все
    //keyTimes = {}; // удаляет все
}

var checked_ = false;

function send_test(self) {
    try {
        checked_  = document.getElementById("test_user").checked;
    } catch(e) {}
    let value_pure = '';
    for (var i = 0; i < arr.length; i++) {
        value_pure += arr[i].key_name;
    }
    if (text_input.innerText.replace(/\s+/g, ' ').trim() === value_pure.replace(/\s+/g, ' ').trim()) {
        console.log(text_input.innerText.replace(/\s+/g, ' ').trim(), "<---->", value_pure.replace(/\s+/g, ' ').trim());
        
        //-------------------------------------------->
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
        //-------------------------------------------->      
    }
    
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
                    bp = document.getElementById("block_post");
                    textOriginal = bp.textContent;
                    
                    console.log("........",bp);
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
//function TESTKEY(self) {
//    console.log(self)
//}

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
    setInterval(recording_key, 1000);
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
//    show_value.removeChild(t_el);
    var message_data = JSON.parse(data.data);
    if (message_data["status"] == "send_test") {
        show_value.removeChild(t_el);
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
        show_value.removeChild(t_el);
        blockup.innerHTML = `<div id="node">
                                        <br>
                                        ${message_data["html"]}
                                        <br>
                                        <button onclick="close_div()">close</button>
                                    </div>`        
        blockup.style.display = "block";
        document.body.style.overflow = 'hidden';
    } else if (message_data["status"]=="Done") { 
        if (message_data["result"]=="Done") {
            result_ = "stop";
        } 
        blockup.innerHTML = `<div id="node">
                                        <div id="text_msg">${message_data["html"]}</div>
                                        <br>
                                        <button type='button' class='Button' onclick="close_div()">close</button>
                                    </div>`        
        blockup.style.display = "block";
        document.body.style.overflow = 'hidden';
    } else if (message_data["status"]=="Error") { 
        blockup.innerHTML = `<div id="node">
                                        <div id="text_msg">${message_data["html"]}</div>
                                        <br>
                                        <button type='button' class='Button' onclick="close_div()">close</button>
                                    </div>`        
        blockup.style.display = "block";
        document.body.style.overflow = 'hidden';
    }
        
};

// регистрация шаг 2
function send_for_reg(self) {
    var crsv = getCookie('csrftoken'); // токен
    let data = JSON.stringify({'KEYPRESS': arr,
                               'text':text_input.innerText}); 
    console.log("SEND_FOR_REG", crsv, data);   
    var http = createRequestObject();
    var linkfull = '/registrationend/';
    if (http) {
        http.open('post', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.setRequestHeader('X-CSRFToken', crsv);
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
                if (http.status == 200) {
                    window.location.replace("/");
                }
            }
        }
        let data = JSON.stringify({'KEYPRESS': arr,
                                   'text':text_input.innerText});
                        
                                   
        http.send(data);  
    }
}

// вход шаг 2
//function send_for_log(self) {
//    var crsv = getCookie('csrftoken'); // токен
//    let data = JSON.stringify({'KEYPRESS': arr,
//                               'text':text_input.innerText}); 
//    // console.log("SEND_FOR_REG", crsv, data);   
//    var http = createRequestObject();
//    var linkfull = '/loginend/';
//    if (http) {
//        http.open('post', linkfull);
//        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
//        http.setRequestHeader('X-CSRFToken', crsv);
//        http.onreadystatechange = function () {
//            if (http.readyState == 4) {
//                if (http.status == 200) {
//                    document.getElementById("show_value").removeChild(t_el);
//                    var data = JSON.parse(http.responseText);
//                    document.getElementById("show_value").innerHTML = `<a href="/">HOME PAGE ${data["user"]}</a>`;
//                    document.getElementById("blockup").innerHTML = `<div id="node">
//                                    <br>
//                                    ${data["html"]}
//                                    <br>
//                                    <button onclick="close_div()">close</button>
//                                </div>`
//                    //'<div id="node">' + http.responseText + '<a onclick="close_div()">закрыть</a></div>';
//                    document.getElementById("blockup").style.display = "block";
//                    document.body.style.overflow = 'hidden';
//                    //window.location.replace("/");
//                }
//            }
//        }
//        let data = JSON.stringify({'KEYPRESS': arr,
//                                   'KEYPRESS_BAD': arr_bad,
//                                   'text':text_input.innerText});
//        document.getElementById("show_value").appendChild(t_el);
//        let value_pure = '';
//        for (var i = 0; i < arr.length; i++) {
//            value_pure += arr[i].key_name;
//        }
//        if (text_input.innerText.replace(/\s+/g, ' ').trim() === value_pure.replace(/\s+/g, ' ').trim()) {
//            console.log(arr, text_input.innerText, value_pure, idx_arr)
//            console.log(text_input.innerText.replace(/\s+/g, ' ').trim(), "<---->", value_pure.replace(/\s+/g, ' ').trim());
//            http.send(data);  
//        }
//    }
//}


//--------------------------------------->
function send_for_log(self) {
    var crsv = getCookie('csrftoken'); // токен
    let data = JSON.stringify({'KEYPRESS': arr,
                               'text':text_input.innerText}); 
    // console.log("SEND_FOR_REG", crsv, data);   
    var http = createRequestObject();
    var linkfull = '/loginend/';
    if (http) {
        http.open('post', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.setRequestHeader('X-CSRFToken', crsv);
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
                if (http.status == 200) {
                    document.getElementById("show_value").removeChild(t_el);
                    var data = JSON.parse(http.responseText);
                    document.getElementById("show_value").innerHTML = `<a href="/">HOME PAGE ${data["user"]}</a>`;
                    document.getElementById("blockup").innerHTML = `<div id="node">
                                    <br>
                                    ${data["html"]}
                                    <br>
                                    <button onclick="close_div()">close</button>
                                </div>`
                    //'<div id="node">' + http.responseText + '<a onclick="close_div()">закрыть</a></div>';
                    document.getElementById("blockup").style.display = "block";
                    document.body.style.overflow = 'hidden';
                    //window.location.replace("/");
                }
            }
        }
        let data = JSON.stringify({'KEYPRESS': arr,
                                   'KEYPRESS_BAD': arr_bad,
                                   'text':text_input.innerText});
        document.getElementById("show_value").appendChild(t_el);
        http.send(data); 
    }
}



