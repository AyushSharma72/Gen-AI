from langchain_text_splitters import  RecursiveCharacterTextSplitter,Language

text="""
class Animal {
    String name;

    Animal(String name) {
        this.name = name;
    }

    void sound() {
        System.out.println(name + " makes a sound");
    }
}

class Dog extends Animal {
    Dog(String name) {
        super(name);
    }

    void bark() {
        System.out.println(name + " barks");
    }
}

public class Main {
    public static void main(String[] args) {
        Dog dog = new Dog("Bruno");
        dog.sound();
        dog.bark();
    }
}"""
 
splitter = RecursiveCharacterTextSplitter.from_language(  # to split python code it uses different types of splitters according to language
    chunk_size=200,
    chunk_overlap=0,
    language=Language.JAVA  # split based on java lang
)

chunks = splitter.split_text(text) #chunks is a list of items for this piece of code i have three items

print(chunks[0])  