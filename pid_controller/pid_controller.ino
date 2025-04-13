// Motor control pins
#define MOTOR_PWM  9  // PWM pin for motor speed
#define MOTOR_IN1  7  // Motor direction pin 1
#define MOTOR_IN2  8  // Motor direction pin 2

// Encoder pins
#define ENCODER_A 2  // Encoder A signal (interrupt pin)
#define ENCODER_B 3  // Encoder B signal

volatile int encoderCount = 0;  // Stores encoder counts

void setup() {
    Serial.begin(9600);  // ✅ Start serial communication
    pinMode(MOTOR_PWM, OUTPUT);
    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    
    pinMode(ENCODER_A, INPUT_PULLUP);
    pinMode(ENCODER_B, INPUT_PULLUP);

    // ✅ Attach interrupt for encoder counting
    attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderISR, RISING);
    
    Serial.println("Setup Complete. Send 'f' to move forward, 'b' for backward, 's' to stop.");
}

void loop() {
    // ✅ Test motor at different speeds and check encoder counts
    testMotorSpeed(100); // 40% speed
    delay(2000);
    testMotorSpeed(200); // 80% speed
    delay(2000);

    // ✅ Print final encoder count after tests
    Serial.print("Final Encoder Count: ");
    Serial.println(encoderCount);

    // ✅ Read user commands for manual testing
    if (Serial.available()) {
        char command = Serial.read();
        handleCommand(command);
    }
}

// ✅ Interrupt Service Routine (ISR) for Encoder
void encoderISR() {
    if (digitalRead(ENCODER_B) == HIGH) {
        encoderCount++;  // Clockwise rotation
    } else {
        encoderCount--;  // Counterclockwise rotation
    }
}

// ✅ Function to test motor at different speeds
void testMotorSpeed(int speed) {
    Serial.print("Testing at Speed: ");
    Serial.println(speed);

    encoderCount = 0; // Reset encoder count
    setMotorSpeed(speed);
    delay(3000);  // Run motor for 3 seconds
    setMotorSpeed(0);

    Serial.print("Encoder Count at ");
    Serial.print(speed);
    Serial.print(": ");
    Serial.println(encoderCount);
}

// ✅ Function to control motor speed & direction
void setMotorSpeed(int speed) {
    if (speed > 0) {  // Forward
        digitalWrite(MOTOR_IN1, HIGH);
        digitalWrite(MOTOR_IN2, LOW);
    } else if (speed < 0) {  // Reverse
        digitalWrite(MOTOR_IN1, LOW);
        digitalWrite(MOTOR_IN2, HIGH);
        speed = -speed;  // Convert to positive for PWM
    } else {  // Stop
        digitalWrite(MOTOR_IN1, LOW);
        digitalWrite(MOTOR_IN2, LOW);
    }
    analogWrite(MOTOR_PWM, speed);  // Set PWM speed (0-255)
}

// ✅ Function to handle serial commands for manual testing
void handleCommand(char command) {
    if (command == 'f') {
        Serial.println("Moving Forward...");
        setMotorSpeed(150);  // 60% speed
    } else if (command == 'b') {
        Serial.println("Moving Backward...");
        setMotorSpeed(-150); // 60% reverse
    } else if (command == 's') {
        Serial.println("Stopping...");
        setMotorSpeed(0);
    } else {
        Serial.println("Invalid command! Use 'f', 'b', 's'.");
    }
}
