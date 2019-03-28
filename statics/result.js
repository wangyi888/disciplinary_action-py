$(document).ready(function(){
$('#sendBtn').on('click', function () {
    console.log('呵呵哈哈哈')
    $('#result').html("<h1>违纪行为识别结果</h1>")
    var content = $('#content').val();
    console.log(content)
    var type = $('#anyou_type').val();

    setTimeout(answers(content,type), 1000);

})
});

function answers(content,type) {

    $.ajax({
        url: "/predict",
        type: "post",
        dataType: "json",
        data: $('#form').serialize(),
        success: function (data) {
            //console.log(data)
            var answer = '<br><h2>识别结果为</h2>'+'<table border="1"><tr>'
            answer += '<th>序号</th><th>违纪行为</th><th>概率</th></tr>'
            for(i=0;i<data.res.length;i++){
                answer += '<tr><td>'+(i+1)+'</td><td>'+data.res[i].label+'</td><td>'+data.res[i].prob+'</td></tr>'
            }
//            console.log(answer)

            $('#result').append(answer);
        }
    });
}



