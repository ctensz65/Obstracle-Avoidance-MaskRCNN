#include <util/atomic.h>
#include <string.h>

/* =========================================== */
/* =========================================== */
/* =============== Cinefa-BOT ================ */
/* =========================================== */
/* =========================================== */

/* =========================================== */
/* PIN OUT ARDUINO */
/* =========================================== */
/* Pins M1 */
#define   ENCA    21
#define   ENCB    19
#define   PWMA    5
#define   IN1     6
#define   IN2     7

/* Pins M2 */
#define   ENCC    20
#define   ENCD    18
#define   PWMB    8
#define   IN3     9
#define   IN4     10

/* Pins M3 */
#define   ENC3_A  3
#define   ENC3_B  4
#define   PWMC    23
#define   IN5     25
#define   IN6     27

/* Pins M4 */
#define   ENC4_A  2
#define   ENC4_B  14
#define   PWMD    33
#define   IN7     29
#define   IN8     31

#define NMOTOR 2

/* =========================================== */
/* Variabel untuk PID */
/* =========================================== */
int posPrev_M1 = 0;
int posPrev_M2 = 0;
long prevT_M1 = 0;
long prevT_M2 = 0;
  
float errorVal[] = {0,0};
float prev_error = 0;
float targetPos[] = {0,0};

/* Volatile directive untuk interupts */
volatile int pos_right = 0;
volatile int pos_left = 0;
volatile float velocity_right = 0;
volatile float velocity_left = 0;
volatile long prevT_left = 0;
volatile long prevT_right = 0;

/* Filter kecepatan dari Noise */
float v1Filt = 0;
float v1Prev = 0;
float v2Filt = 0;
float v2Prev = 0;
float v3Filt = 0;
float v3Prev = 0;
float v4Filt = 0;
float v4Prev = 0;

/* Untuk Kalkulasi PID */
float eintegral[] = {0,0};
float ederivative = 0;

/* PID Value */
float kp = 4;
float ki = 9;
float kd = 2000;
float output;

int dir;
int pwr;
int normalMotor = 150;
/* =========================================== */
/* Variabel untuk Serial Communication */
/* =========================================== */

/* Keperluan Serial Communication */
const byte NumChars = 32;
char ReceivedChars[NumChars];
char tempChars[NumChars];

boolean NewData = false;
char arahMessage[NumChars] = {0};
int valuePython = 0;

unsigned long curMillis;

void setup() {
  Serial.begin(115200);

  /* MOTOR RIGHT M1 */
  pinMode(ENCA,INPUT);
  pinMode(ENCB,INPUT);
  pinMode(PWMA,OUTPUT);
  pinMode(IN1,OUTPUT);
  pinMode(IN2,OUTPUT);

  attachInterrupt(digitalPinToInterrupt(ENCA),
                  readEncoderM1,RISING);

  /* MOTOR RIGHT M2 */
  pinMode(ENCC,INPUT);
  pinMode(ENCD,INPUT);
  pinMode(PWMB,OUTPUT);
  pinMode(IN3,OUTPUT);
  pinMode(IN4,OUTPUT);

  attachInterrupt(digitalPinToInterrupt(ENCC),
                  readEncoderM2,RISING);

  /* MOTOR RIGHT M3 */
  pinMode(ENC3_A,INPUT);
  pinMode(ENC3_B,INPUT);
  pinMode(PWMC,OUTPUT);
  pinMode(IN5,OUTPUT);
  pinMode(IN6,OUTPUT);

  attachInterrupt(digitalPinToInterrupt(ENC3_A),
                  readEncoderM2,RISING);
                  
  /* MOTOR RIGHT M4 */
  pinMode(ENC4_A,INPUT);
  pinMode(ENC4_B,INPUT);
  pinMode(PWMD,OUTPUT);
  pinMode(IN7,OUTPUT);
  pinMode(IN8,OUTPUT);

  attachInterrupt(digitalPinToInterrupt(ENC4_A),
                  readEncoderM2,RISING);
                  
  setMotorStop();
}

void loop() {
  curMillis = millis();
  RecvWithStartEndMarkers();
  // Set the motor speed and direction
  if (NewData == true) {
      strcpy(tempChars, ReceivedChars);
          // this temporary copy is necessary to protect the original data
          //   because strtok() used in parseData() replaces the commas with \0
      ParseData();
      replyToPC();
      NewData = false;
  }
  
  calculatePID();

  dir = 1;
  if (output < 0){
    dir = -1;
  }
  pwr = (int) fabs(output);
  
  if(pwr > 255){
    pwr = 255;
  }
  if(pwr < 40 && errorVal != 0){
    pwr = 40;
  }
  
  if (!strcmp(arahMessage, "kiri")){
    setMotorRight(dir,pwr);
  }
  else if (!strcmp(arahMessage, "kanan")){
    setMotorLeft(dir,pwr);
  }
  else if (!strcmp(arahMessage, "lurus")){
    setMotorNormal();
  }
  else if (!strcmp(arahMessage, "stop")){
    setMotorStop();
  }

  /*
  Serial.print(targetPos);
  Serial.print(" ");
  Serial.print(v1Filt);
  Serial.println();
  delay(1);
  */
}

/* =========================================== */
/* Processing Serial Data from Python */
/* =========================================== */

void RecvWithStartEndMarkers()
{
    static boolean RecvInProgress = false;
    static byte ndx = 0;      // index
    char StartMarker = '<';
    char EndMarker = '>';
    char rc;          // received data

    while (Serial.available() > 0 && NewData == false)
    {
        rc = Serial.read();               // test for received data

        if (RecvInProgress == true)
        {         // found some!!
            if (rc != EndMarker)          // <> end marker
            {
                ReceivedChars[ndx] = rc;  // 1st array position=data
                ndx++;                    // next index 
                if (ndx >= NumChars)      // if index>= number of chars
                { 
                    ndx = NumChars - 1;   // index -1
                }
            }
            else                          // end marker found
            {
                ReceivedChars[ndx] = '\0'; // terminate the string  
                RecvInProgress = false;
                ndx = 0;                  // reset index
                NewData = true;           // new data received flag
            }
        }

        else if (rc == StartMarker)       // signal start of new data
        {
            RecvInProgress = true;
        }
    }
}


void ParseData()                          // split the data into its parts
{                                         // Serial.println(MessageFromPC); ECHO data received
    char * StrTokIndx;                    // this is used by strtok() as an index

    StrTokIndx = strtok(tempChars,",");   // get the first control word
    strcpy(arahMessage, StrTokIndx);  // copy it to messageFromPC
    
    StrTokIndx = strtok(NULL, ",");       // this continues after 1st ',' in the previous call
    valuePython = atoi(StrTokIndx);    // convert this part to the first integer
}

void replyToPC() {
    Serial.print("<Arah:");
    Serial.print(arahMessage);
    Serial.print(" vMI:");
    Serial.print(v1Filt);
    Serial.print(" vM2:");
    Serial.print(v3Filt);
    Serial.print(" Time:");
    Serial.print(curMillis >> 9); // divide by 512 is approx = half-seconds
    Serial.println(">");
  }

/* =========================================== */
/* Proccessing Value Motor with PID Controller */
/* =========================================== */

void calculatePID(){
  // read the position in an atomic block
  // to avoid potential misreads
  int posM1 = 0;
  int posM2 = 0;
  
  float velocity2_M1 = 0;
  float velocity2_M2 = 0;
  
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE){
    posM1 = pos_right;
    posM2 = pos_left;
    velocity2_M1 = velocity_right;
    velocity2_M2 = velocity_left;
  }

  // Compute velocity with method 1 (Motor Kanan)
  long currT = micros();
  float deltaT_M1 = ((float) (currT-prevT_M1))/1.0e6;
  float velocity1_M1 = (posM1 - posPrev_M1)/deltaT_M1;
  posPrev_M1 = posM1;
  prevT_M1 = currT;
  
  // Compute velocity with method 1 (Motor Kiri)
  float deltaT_M2 = ((float) (currT-prevT_M2))/1.0e6;
  float velocity1_M2 = (posM2 - posPrev_M2)/deltaT_M2;
  posPrev_M2 = posM2;
  prevT_M2 = currT;

  // Convert count/s to RPM
  float v1 = velocity1_M1/600.0*60.0;
  float v2 = velocity2_M1/600.0*60.0;

  float v3 = velocity1_M2/600.0*60.0;
  float v4 = velocity2_M2/600.0*60.0;

  // Low-pass filter (25 Hz cutoff)
  v1Filt = 0.854*v1Filt + 0.0728*v1 + 0.0728*v1Prev; // new speed
  v1Prev = v1;
  v2Filt = 0.854*v2Filt + 0.0728*v2 + 0.0728*v2Prev;
  v2Prev = v2;

  v3Filt = 0.854*v3Filt + 0.0728*v3 + 0.0728*v3Prev; // new speed
  v3Prev = v3;
  v4Filt = 0.854*v4Filt + 0.0728*v4 + 0.0728*v4Prev;
  v4Prev = v4;

  // Set a target (setpoint)
  targetPos[0] = float(valuePython);
  targetPos[1] = targetPos[0]*-1;
  
  if (!strcmp(arahMessage, "kiri")){
    errorVal[0] = targetPos[0]-v1Filt;
    eintegral[0] = eintegral[0] + errorVal[0]*deltaT_M1;
    // ederivative = (errorVal - prev_error) / deltaT;

    output = kp*errorVal[0] + ki*eintegral[0];
  }
  else if (!strcmp(arahMessage, "kanan")){
    errorVal[1] = targetPos[1]-v3Filt;
    eintegral[1] = eintegral[1] + errorVal[1]*deltaT_M2;

    output = kp*errorVal[1] + ki*eintegral[1];
  }
}

/* =========================================== */
/* Inisialisasi Motor */
/* =========================================== */

void setMotorRight(int dir, int pwmVal){
  analogWrite(PWMB,0);
  analogWrite(PWMC,0); 
  if(dir == 1){ 
    digitalWrite(IN1,HIGH);
    digitalWrite(IN2,LOW);
    analogWrite(PWMA,pwmVal);

    digitalWrite(IN7,LOW);
    digitalWrite(IN8,HIGH);
    analogWrite(PWMD,pwmVal);
  }
  else if(dir == -1){
    digitalWrite(IN1,LOW);
    digitalWrite(IN2,HIGH);
    analogWrite(PWMA,pwmVal);

    digitalWrite(IN7,HIGH);
    digitalWrite(IN8,LOW);
    analogWrite(PWMD,pwmVal);
  }
  else{
    // Or dont turn
    digitalWrite(IN1,LOW);
    digitalWrite(IN2,LOW); 
  }
}

void setMotorLeft(int dir, int pwmVal){
  analogWrite(PWMA,0);
  analogWrite(PWMD,0);
  if(dir == 1){ 
    // M2
    digitalWrite(IN3,HIGH);
    digitalWrite(IN4,LOW);
    analogWrite(PWMB,pwmVal);
    
    digitalWrite(IN5,LOW);
    digitalWrite(IN6,HIGH);
    analogWrite(PWMC,pwmVal);
  }
  else if(dir == -1){
    // M2
    digitalWrite(IN3,LOW);
    digitalWrite(IN4,HIGH);
    analogWrite(PWMB,pwmVal);
    
    digitalWrite(IN5,HIGH);
    digitalWrite(IN6,LOW);
    analogWrite(PWMC,pwmVal);
  }
  else{
    // Or dont turn
    digitalWrite(IN3,LOW);
    digitalWrite(IN4,LOW); 
  }
}

void setMotorNormal(){
  digitalWrite(IN1, HIGH); 
  digitalWrite(IN2, LOW);
  analogWrite(PWMA,normalMotor);
  
  digitalWrite(IN3, LOW); 
  digitalWrite(IN4, HIGH);
  analogWrite(PWMB,normalMotor);
  
  digitalWrite(IN5, HIGH); 
  digitalWrite(IN6, LOW);
  analogWrite(PWMC,normalMotor);
  
  digitalWrite(IN7, LOW); 
  digitalWrite(IN8, HIGH);
  analogWrite(PWMD,normalMotor);
}

void setMotorStop(){
  analogWrite(PWMA,0);
  analogWrite(PWMB,0);
  analogWrite(PWMC,0);
  analogWrite(PWMD,0);
}

/* =========================================== */
/* Read Encoder & Speed */
/* =========================================== */

void readEncoderM1(){ // right
  // Read encoder B when ENCA rises
  int encB = digitalRead(ENCB);
  int increment = 0;
  if(encB > 0){
    // If B is high, increment forward
    increment = 1;
  }
  else{
    // Otherwise, increment backward
    increment = -1;
  }
  pos_right = pos_right + increment;

  // Compute velocity with method 2
  long currT = micros();
  float deltaT = ((float) (currT - prevT_right))/1.0e6;
  velocity_right = increment/deltaT;
  prevT_right = currT;
}

void readEncoderM2(){ // left
  // Read encoder B when ENCA rises
  int encD = digitalRead(ENCD);
  int increment = 0;
  if(encD > 0){
    // If B is high, increment forward
    increment = 1;
  }
  else{
    // Otherwise, increment backward
    increment = -1;
  }
  pos_left = pos_left + increment;

  // Compute velocity with method 2
  long currT = micros();
  float deltaT = ((float) (currT - prevT_left))/1.0e6;
  velocity_left = increment/deltaT;
  prevT_left = currT;
}
