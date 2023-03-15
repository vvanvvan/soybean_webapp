import 'dart:html';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:soybean_forecast/main.dart';
import 'API.dart';
import 'dart:convert';

import 'dart:io';
import 'package:path/path.dart';
import 'package:excel/excel.dart';


class TrainPage extends StatelessWidget {
  const TrainPage({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {

    return MaterialApp(
      title: 'train interface',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        canvasColor: Colors.amber.shade300,
      ),
      home: const MyHomePage(title: 'train interface'),


    );
  }
}


class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);
  final String title;
  @override
  State<MyHomePage> createState() => _MyHomePageState();

}

class _MyHomePageState extends State<MyHomePage> {
  @override
  String month_fromexcel = " ";

  void initState()  {
    super.initState();
    _excel_row();
  }

  _excel_row() async {
    url = "http://127.0.0.1:5000/excel_value";
    Data = await Getdata(url);
    var DecodedData = jsonDecode(Data);
    setState(() {
      month_fromexcel = DecodedData.toString();
      final splitted = month_fromexcel.split(' ');
      var MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"];
      for(int i=0; i<MONTHS.length;++i){
        if(splitted[0]==MONTHS[i]){
          valueMonth = i+1;
        }
      }

      valueYear=  int.parse(splitted[1]);
    });

  }
  var validdrop = Text ("โปรดเลือกเดือน",
    // overflow: TextOverflow.fade,
    style:GoogleFonts.mitr(
      textStyle: TextStyle(
          color: Colors.black,
          fontSize: 16.0),
    ),
  );

  int _selectedIndex = 0;
  final _formKey = GlobalKey<FormState>();
  final _text1 = TextEditingController();  //textediting
  final _text2 = TextEditingController();// controller
  bool _validate1 = false;
  bool _validate2= false;//variable to store the bool value
  bool _validate3= false;


  String? excelTH ="เมษายน 2565";

  int index=0;
  String? selectedValue;
  String url='';
  String year='';
  var Data;
  String QueryText = '';

  late int valueMonth;
  late int valueYear;
  String valueThai='';
  String nameImgTrain = 'cupertino_activity.gif';
  String textTrain = 'กำลังเทรนโมเดล กรุณารอสักครู่';
  String titleTrain ='รอสักครู่';


  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
      print( _selectedIndex);
      if( _selectedIndex == 0){
        Navigator.of(this.context).push(MaterialPageRoute(builder: (context) => MyApp()));
      }
      // else if( _selectedIndex == 1){
      //   Navigator.of(this.context).push(MaterialPageRoute(builder: (context) => TrainPage(text: ,))); //Navigator.of(context).push(MaterialPageRoute(builder: (context) => TrainPage()));
      //
      // }
    });
  }
  void dispose1() {
    _text1.dispose();
    super.dispose();
  }
  void dispose2() {
    _text2.dispose();
    super.dispose();
  }
  void showDialogUpdate(BuildContext context) async {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title:
        Text(
            "ส่งค่าสำเร็จ",
            style: GoogleFonts.mitr(
              textStyle: TextStyle(
                  color: Colors.black,
                  fontSize: 30.0),
            ),
            textAlign: TextAlign.center

        ),
        content: Stack(
          alignment: Alignment.center,
          children: [
            Container(
                alignment: Alignment.center,
                width: 300, height: 200,
                child: Image.asset('asset/images/check-correct.gif', height: 270, width: 270,)),
          ],
        ),
        actions: [
          TextButton(
              child: Text("ตกลง",
                style: GoogleFonts.mitr(
                  textStyle: TextStyle(
                      color: Colors.green,
                      fontSize: 22.0),
                ),
              ),
              onPressed: () => {Navigator.pop(context)}
          ),
        ],

      ),
    );

  }

  void showDialogTrain(BuildContext contextUpdate) async {
    showDialog(
      context: contextUpdate,
      builder: (contextUpdate) => AlertDialog(

        title:
        Text(titleTrain,
            style: GoogleFonts.mitr(
              textStyle: TextStyle(
                  color: Colors.black,
                  fontSize: 30.0),
            ),
            textAlign: TextAlign.center

        ),
        content: Container(
          width: 300,
          height: 130,
          decoration: BoxDecoration(
            // borderRadius: BorderRadius.circular(15),
            image: DecorationImage(
              image: AssetImage('asset/images/$nameImgTrain'),
              fit: BoxFit.contain,
            ),
          ),
          child: Text(textTrain,
            textAlign: TextAlign.center,
            style: GoogleFonts.mitr(
              textStyle: TextStyle(
                  color: Colors.black,
                  fontSize: 20.0),
            ),
          ),

        ),
        actions: [
          TextButton(
            child: Text("ตกลง",
              style: GoogleFonts.mitr(
                textStyle: TextStyle(
                    color: Colors.green,
                    fontSize: 22.0),
              ),
            ),

            onPressed: (){
              titleTrain = 'รอสักครู่';
              textTrain = 'กำลังเทรนโมเดล กรุณารอสักครู่';
              nameImgTrain = 'cupertino_activity.gif';
              Navigator.pop(contextUpdate);

            },

          ),
        ],

      ),
    );

  }

  @override
  Widget build(BuildContext context) {
    // final height = MediaQuery.of(context).size.height;
    // final width = MediaQuery.of(context).size.width;
    var now = new DateTime.now();
    var MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"];

    final List<String> items = [];
    String formattedDateTime(int num) {
      DateTime now = new DateTime.now();
      var years = now.year+543;
      return MONTHS[now.month+num]+" "+years.toString();
    }

    String formattedDateOver(int num) {
      int count = now.year+543;
      // var new_year = new DateTime(2567, num);
      var year = count+1;
      // print("เดือน $num");
      //print(MONTHS[num]);
      return MONTHS[num-1]+" "+year.toString();
    }


    int count = now.month;
    int countYear = 1;
    var count_month=12;

    for(int i=0;i<count_month;++i) {
      count++;
      // print(count);

      if(count > 12){
        items.add(formattedDateOver(countYear));
        // print(countYear);
        countYear++;
      }else{
        items.add(formattedDateTime(i));

      }
    }

    return Scaffold(
      // appBar: AppBar(
      // title: Text('Soybean Forecast'),
      // ),
      backgroundColor: Color.fromRGBO(255, 194, 0, 0.15),
      resizeToAvoidBottomInset: false,
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            scrollDirection: Axis.vertical,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Container(
                    child: Row(
                      children: [
                        Image.asset('asset/images/logo.png',
                          fit: BoxFit.fill,
                          height: 55,
                        ),
                        Expanded(
                          child: Text(
                            'เพิ่มข้อมูลราคากากถั่วเหลืองนำเข้าไทย',
                            textAlign: TextAlign.start,
                            style: GoogleFonts.mitr(
                              textStyle: TextStyle(
                                  color: Colors.black54,
                                  fontSize: 25.0),
                            ),
                          ),
                        ),
                      ],
                    ),
                    decoration: new BoxDecoration(
                      color: Colors.amber.shade300,
                      boxShadow: [new BoxShadow(blurRadius: 3.0)],
                      borderRadius: BorderRadius.vertical(bottom: Radius.circular(35.0)),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(top: 12.0),
                  child: SizedBox(
                    child: Row(
                      children: [
                        Expanded(
                            child: Container(
                              // alignment: Alignment.center,
                              height: 200,
                              margin: const EdgeInsets.symmetric(horizontal: 20),
                              decoration: BoxDecoration(
                                  image: const DecorationImage(
                                      image: AssetImage('asset/images/banner2.png'),
                                      fit: BoxFit.cover
                                  ),
                                  borderRadius: BorderRadius.circular(30)
                              ),
                            )
                        ),
                      ],
                    ),
                  ),
                ),

                Padding(
                  padding: const EdgeInsets.only(top: 20.0),
                  child:  Container(
                    alignment: Alignment.bottomCenter,
                    width: 663,
                    height: 306,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      color: Colors.amber.shade300,
                    ),
                    child: Padding(
                      padding: const EdgeInsets.only(top:35.0, left:35.0, right: 35.0),
                      child: Column(
                        children: [
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 8.0),
                            child: Row(
                                children: [
                                  Container(
                                    width: 230,
                                    child: Text('เดือนที่ต้องเพิ่มข้อมูล ',style: GoogleFonts.mitr(
                                      textStyle: TextStyle(
                                          color: Colors.black87,
                                          fontSize: 18.0),
                                    ),
                                    ),
                                  ),
                                  SizedBox(width: 80.0,), //35
                                  SizedBox(
                                    // decoration: BoxDecoration(color: Colors.grey.shade50,),
                                    height: 70,
                                    width: 150.0,
                                    child: TextField(
                                      controller: _text1,
                                      onChanged: (value) {
                                        setState(() {
                                          excelTH = value;
                                          _text1.text.isEmpty ? _validate1 = true : _validate1 = false;
                                        });
                                      },
                                      decoration: InputDecoration(
                                        hintText:  month_fromexcel,
                                        hintStyle: GoogleFonts.mitr(
                                          textStyle: TextStyle(
                                              color: Colors.blue,
                                              fontSize: 18.0),
                                        ),
                                        border: InputBorder.none,
                                      ),
                                      style: GoogleFonts.mitr(
                                        textStyle: TextStyle(
                                            color: Colors.black,
                                            fontSize: 18.0),
                                      ),
                                      textAlign: TextAlign.center,
                                      readOnly: true,

                                    ),
                                  ),
                                ]
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.only(left:8, right:8, top: 22), //horizontal: 8.0, vertical: 22
                            child: Row(
                              children: [
                                Container(
                                  width: 250,
                                  child: Expanded(
                                    child: Text('กรอกราคากากถั่วเหลืองนำเข้าไทย',style: GoogleFonts.mitr(
                                      textStyle: TextStyle(
                                          color: Colors.black87,
                                          fontSize: 18.0),
                                    ),
                                    ),
                                  ),
                                ),
                                SizedBox(width: 60.0,),
                                FractionalTranslation(
                                  translation: Offset(0,0),
                                  child: SizedBox(
                                    height: 70,
                                    width: 150.0,
                                    child: TextField(
                                      inputFormatters: [
                                        FilteringTextInputFormatter.allow(RegExp(r'^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)')),
                                        FilteringTextInputFormatter.deny(RegExp(r'^0+'), ),],
                                      style: GoogleFonts.mitr(
                                        textStyle: TextStyle(
                                            color: Colors.black,
                                            fontSize: 18.0),
                                      ),
                                      controller: _text2,
                                      textAlign: TextAlign.center,
                                      decoration: InputDecoration(
                                        labelStyle: GoogleFonts.mitr(
                                          textStyle: TextStyle(
                                              color: Colors.blue,
                                              fontSize: 15.0),
                                        ),
                                        labelText: 'กรอกตัวเลข',
                                        filled: true,
                                        fillColor: Colors.grey.shade50,
                                        errorText: _validate2 ? 'โปรดกรอกช่องนี้' : null,
                                        // errorStyle: GoogleFonts.mitr(
                                        //   textStyle: TextStyle(
                                        //       color: Colors.red,
                                        //       fontSize: 12.0),
                                        // ),
                                      ),
                                      onChanged: (value) {
                                        valueThai=value;
                                        setState(() {
                                          _text2.text.isEmpty ? _validate2 = true : _validate2 = false;
                                        });
                                      },
                                    ),
                                  ),
                                ),
                                SizedBox(width: 15.0,),
                                Container(
                                  child: Text('บาท/กิโลกรัม',style: GoogleFonts.mitr(
                                    textStyle: TextStyle(
                                        color: Colors.black87,
                                        fontSize: 18.0),
                                  ),
                                  ),
                                ),
                              ],
                            ),
                          ),

                          Padding(
                            padding: const EdgeInsets.only(top: 17, bottom: 18),
                            child: Expanded(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  Container(
                                    height: 60,
                                    alignment: Alignment.center,
                                    child: SizedBox(
                                      height: 50,
                                      width: 160,
                                      child: ElevatedButton(
                                        style: ElevatedButton.styleFrom(
                                          primary: Colors.green,
                                          // textStyle: TextStyle(fontSize: 20),
                                          shape: RoundedRectangleBorder(
                                            borderRadius: BorderRadius.circular(10.0),
                                          ),
                                        ),
                                        onPressed: () async {

                                          // _text1.text.isEmpty ? _validate1 = true : _validate1 = false;
                                          _text2.text.isEmpty ? _validate2 = true : _validate2= false;

                                          if(_validate2==false) {
                                            // month_fromexcel
                                            // const string = 'Hello world!';
                                            // [Hello, world!];

                                            final splitted = month_fromexcel.split(' ');
                                            print('dddd');
                                            print(splitted[0]);

                                            print('monthttt');
                                            print(month_fromexcel);
                                            url = "http://127.0.0.1:5000/update_data?New_Month=$valueMonth&Year=$valueYear&priceThai=$valueThai";
                                            print(url);
                                            Data = await Getdata(url);
                                            var DecodedData = jsonDecode(Data);
                                            print('DecodedData $DecodedData');
                                            print('Data $Data');


                                            if(Data!='') {
                                              showDialogUpdate(context);
                                            }
                                            setState(() {
                                              QueryText = DecodedData.toString();

                                            });

                                          }else{
                                            setState(() {
                                              // _text1.text.isEmpty ? _validate1 = true : _validate1 = false;
                                              _text2.text.isEmpty ? _validate2 = true : _validate2= false;

                                            });
                                          }

                                          //  Navigator.of(context).pop();
                                        },
                                        child: Text("อัปเดตข้อมูล" ,style: GoogleFonts.mitr(
                                          textStyle: TextStyle(
                                              color: Colors.white,
                                              fontSize: 18.0),
                                        ),
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),


                  ),
                ),
                SizedBox(height: 20.0,),
                Padding(
                  padding: const EdgeInsets.only(bottom: 20),
                  child: Container(
                    width: 663,
                    height: 160,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      color: Colors.amber.shade300,
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(17.0),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text('กดปุ่มด้านล่างเพื่อเทรนโมเดล',style: GoogleFonts.mitr(
                            textStyle: TextStyle(
                                color: Colors.black87,
                                fontSize: 18.0),
                          ),
                          ),
                          SizedBox(width: 94.0,),
                          Padding(
                            padding: const EdgeInsets.only(top: 15.0),
                            child: Expanded(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [

                                  Container(
                                    height: 60,
                                    alignment: Alignment.center,
                                    child: SizedBox(
                                      height: 50,
                                      width: 150,
                                      child: ElevatedButton(
                                        style: ElevatedButton.styleFrom(
                                          primary: Colors.green,
                                          // textStyle: TextStyle(fontSize: 20),
                                          shape: RoundedRectangleBorder(
                                            borderRadius: BorderRadius.circular(10.0),
                                          ),
                                        ),
                                        onPressed: () async {
                                          // showDialogTrain(context);
                                          url = "http://127.0.0.1:5000/train_model?";
                                          print(url);
                                          Data = await Getdata(url);
                                          var DecodedData = jsonDecode(Data);
                                          print('DecodedData $DecodedData');
                                          print('Data $Data');
                                          setState(() {
                                            if(Data!='') {
                                              titleTrain = "เทรนโมเดลสำเร็จแล้ว";
                                              nameImgTrain  = "check-correct.gif";
                                              textTrain = "";
                                              showDialogTrain(context);
                                            }
                                            QueryText = DecodedData.toString();

                                          });


                                        },
                                        child: Text("เทรนโมเดล" ,style: GoogleFonts.mitr(
                                          textStyle: TextStyle(
                                              color: Colors.white,
                                              fontSize: 18.0),
                                        ),
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),

              ],
            ),
          ),
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.timeline),
            label: 'ทำนายราคา',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.psychology),
            label: 'เทรนโมเดล',
          ),
        ],
        selectedLabelStyle: GoogleFonts.mitr(

          textStyle: TextStyle(
              color: Colors.black,
              fontSize: 20.0),
        ),
        unselectedLabelStyle:  GoogleFonts.mitr(
          textStyle: TextStyle(
              color: Colors.black,
              fontSize: 18.0),
        ),
        selectedItemColor: Colors.black54,
        unselectedItemColor: Colors.green,
        onTap: _onItemTapped,
      ),
    );
  }
}


