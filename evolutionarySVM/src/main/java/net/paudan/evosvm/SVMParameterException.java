package net.paudan.evosvm;

class SVMParameterException extends Exception{

    public SVMParameterException(Throwable cause) {
        super(cause);
    }

    public SVMParameterException(String message, Throwable cause) {
        super(message, cause);
    }

    public SVMParameterException(String message) {
        super(message);
    }

    public SVMParameterException() {
    }


}
