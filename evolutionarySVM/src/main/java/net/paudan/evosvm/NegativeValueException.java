package net.paudan.evosvm;

public class NegativeValueException extends Exception {

    public NegativeValueException(Throwable cause) {
        super(cause);
    }

    public NegativeValueException(String message, Throwable cause) {
        super(message, cause);
    }

    public NegativeValueException(String message) {
        super(message);
    }

    public NegativeValueException() {
    }

}
