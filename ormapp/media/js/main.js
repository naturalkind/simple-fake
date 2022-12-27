var icon_back = document.getElementsByClassName("icon_back");
icon_back[0].style.display = "none";

// USER IDENTIFICATION KEYSTROKE
var idk_block = document.createElement("div");
idk_block.id = "idl-block";
idk_block.style.display = "none";
//<p><b>name: ${document.getElementById("user_info").innerText}</b><br>
//<br>
idk_block.innerHTML = `
                       <h3>Enter this message:</h3>
                       <br>
                       <p style="font-size:20px;">повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим, 
                          как будет изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в
                          которых работает оператор. желательно, чтобы сначала работал в одних и тех же условиях.
                          повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим, как будет
                          изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в которых
                          работает оператор. желательно, чтобы сначала работал в одних и тех же условиях.
                          повторим этот эксперимент несколько раз с одним и тем же оператором и посмотрим, как будет
                          изменяться статистика на этом коротком тесте. обязательно фиксируем условия, в которых
                          работает оператор. желательно, чтобы сначала работал в одних и тех же условиях.</p> 
                       <div id="text_input" class="message_textarea" role="textbox" contenteditable="true" aria-multiline="true" aria-required="true" style="background: white;font-size: 26px;margin: 9px auto;"></div>
                       <button type="button" onclick="send_test(this)" 
                                             indicator="send" 
                                             class="Button"
                                             style="margin: 4px auto; display: block;">SEND</button>`
// Теория - это когда все известно, но ничего не работает                       
// <button type="button" onclick="send_test()">SEND</button>                       
document.getElementById("testbox").appendChild(idk_block);

var text_input = document.getElementById("text_input");

var keyTimes = {};
var arr = [];
text_input.onkeydown = text_input.onkeyup = text_input.onkeypress = handle;
function handle(e) {
    //console.log("WALL KEYPRESS", e, e.type, keyTimes);
    if (e.type == "keydown") {
        if (!keyTimes["key" + e.which]) {
            keyTimes["key" + e.which] = new Date().getTime();
        }    
    } else if (e.type == "keyup") {
        if (keyTimes["key" + e.which]) {
            var x = new Date().getTime() - keyTimes["key" + e.which];
            //keyTimes["key" + e.which] = {"time_press":x / 1000.0, "key_name":e.key, "key_code":e.keyCode}
            arr.push({"time_press":x / 1000.0, 
                      "key_name":e.key, 
                      "key_code":e.keyCode, 
                      "end_time_press":new Date().getTime()/1000.0,});
            delete keyTimes["key" + e.which];
            //console.log(e.key, x / 1000.0);
        }   
    }
}
function recording_key() {
    console.log('tick', arr)
    ws.send(JSON.stringify({'KEYPRESS': arr}));
    arr.length = 0 // удаляет все
    //arr = [];
    //keyTimes = {};
}


function save_test(self) {
    var NameABC = document.getElementById("Name_ABC").value;
    ws.send(JSON.stringify({'save_test': 'ok', 'NameABC':NameABC}));
    text_input.innerText = "";
} 
function send_test(self) {
    ws.send(JSON.stringify({'send_test': 'ok'}));
}

//----------------------------------->

var reg_bt = document.getElementById("reg_bt");
var log_bt = document.getElementById("log_bt");
var idk_bt = document.getElementById("idk_bt");
reg_bt.style.margin = "4px auto";
log_bt.style.margin = "4px auto";
idk_bt.style.margin = "4px auto";

// Keystroke
idk_bt.addEventListener('click', function(e) {
    idk_block.style.display = "block";
    //registration.style.display = "none";
    // кнопки
    reg_bt.style.display = "none";
    log_bt.style.display = "none";
    idk_bt.style.display = "none";
    // кнопка назад
    icon_back[0].style.display = "block"
    
    document.getElementsByClassName("testbox")[0].style.width = "740px";
    document.getElementById("logo24").style.display = "none";
    // отправлять каждую секунду данные
    //setInterval(recording_key, 1000);
});

//
reg_bt.addEventListener('click', function(e) {
    //registration.style.display = "block";
    reg_bt.style.display = "none";
    log_bt.style.display = "none";
    idk_bt.style.display = "none";
    // кнопка назад
    icon_back[0].style.display = "block"
});


log_bt.addEventListener('click', function(e) {
    reg_bt.style.display = "none";
    log_bt.style.display = "none";
    idk_bt.style.display = "none";
    // кнопка назад    
    icon_back[0].style.display = "block"
});   
// кнопка назад     
icon_back[0].addEventListener('click', function(e) {
    // кнопка назад 
    icon_back[0].style.display = "none"
    // кнопки
    reg_bt.style.display = "block";
    log_bt.style.display = "block";
    idk_bt.style.display = "block";
    
    //registration.style.display = "none";
    idk_block.style.display = "none";
    
    document.getElementsByClassName("testbox")[0].style.width = "343px";
    document.getElementById("logo24").style.display = "block";    
});    


//---------------------------------->

var ws = new WebSocket("ws://178.158.131.41:8998/"); // IP

//---------------------------------->
//function Register() {
//    console.log("Confirm......", 
//                 document.getElementById('name_name').value,
//                 document.getElementById('name_pass').value,
//                 document.getElementById('name_mail').value,
//                 document.getElementById('name_phone').value)
//    ws.send(JSON.stringify({'Register': { 'Name' : document.getElementById('name_name').value,
//                                          'Pass' : document.getElementById('name_pass').value,
//                                          'Phone': document.getElementById('name_phone').value,
//                                          'Mail': document.getElementById('name_mail').value}}));
//}
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
    var crsv = getCookie('csrftoken'); // токен
    console.log(crsv);
    var http = createRequestObject();
    var linkfull = '/registration/';
    if (http) {
        http.open('post', linkfull);
        http.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        http.setRequestHeader('X-CSRFToken', crsv);
        http.onreadystatechange = function () {
            if (http.readyState == 4) {
//                alert("отправлено");
//                console.log(http.responseText);
            }
        }
        if (document.getElementById('name_name').value != "" && document.getElementById('name_pass').value != "") {
            http.send(JSON.stringify({ 'Name' : document.getElementById('name_name').value,
                                       'Pass' : document.getElementById('name_pass').value}));         
        } else {
            alert("заполните поля")
        }
  
    } 
}



                
                
//ws.binaryType = 'arraybuffer';
ws.onopen = function() {
    console.log("connection was established");
};



ws.onmessage = function(data) {
    var message_data = JSON.parse(data.data);
    console.log(message_data)
    if (message_data["switch"] == "SendTest") {
    } else if (message_data["switch"] == "MoreData") {
    } else if (message_data["switch"]=="Done") { 
    }
        
};

