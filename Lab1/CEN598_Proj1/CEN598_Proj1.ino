enum State {
  DARK,
  RED,
  BLUE,
  GREEN
};

State currentState = DARK;

int redPin = 22;
int greenPin = 23;
int bluePin = 24;

// Records when the current state was entered.
// Used to implement non-blocking timed transitions.
unsigned long stateStartTime = 0;


void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);

  Serial.begin(115200);
  while (!Serial) {
  ;
  }

  changeState(DARK);
  Serial.println("State machine started");
  Serial.println("Enter 's' to simulate switch press");
}


void loop() {
  bool switchPressed = false;

  // Check for a virtual switch press from the Serial Monitor.
  // Entering 's' or 'S' represents pressing SW.
  if (Serial.available() > 0) {
    char input = Serial.read();

    if (input == 's' || input == 'S') {
      switchPressed = true;
    }
  }

  switch (currentState) {
    case DARK:
      if (switchPressed) {
        changeState(RED);
      }
      break;

    //I'm using millis here instead of delay because delay will ignore live inputs from my keyboard
    //If I used delays, then it would just be stuck flipping between red and dark
    case RED:
      // SW has priority over timeout
      if (switchPressed) {
        changeState(BLUE);
      } else if (millis() - stateStartTime >= 5000) { //5 sec
        changeState(DARK);
      }
      break;

    case BLUE:
      // SW has priority over timeout
      if (switchPressed) {
        changeState(GREEN);
      } else if (millis() - stateStartTime >= 4000) { //4 sec
        changeState(RED);
      }
      break;

    case GREEN:
      // SW has priority over timeout
      if (switchPressed) {
        changeState(DARK);
      } else if (millis() - stateStartTime >= 3000) { //3 sec
        changeState(BLUE);
      }
      break;
  }
}


// Helper functions

void SetColor(int red, int green, int blue) {
  analogWrite(redPin, red);
  analogWrite(greenPin, green);
  analogWrite(bluePin, blue);
}


void changeState(State newState) {
  currentState = newState;
  stateStartTime = millis();

  switch (currentState) {
    case DARK:
      SetColor(255, 255, 255);
      Serial.println("State: DARK");
      break;

    case RED:
      SetColor(0, 255, 255);
      Serial.println("State: RED");
      break;

    case BLUE:
      SetColor(255, 255, 0);
      Serial.println("State: BLUE");
      break;

    case GREEN:
      SetColor(255, 0, 255);
      Serial.println("State: GREEN");
      break;
  }
}