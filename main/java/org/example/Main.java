package org.example;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Main {
    public static void main(String[] args) {

        Parinte p=new Parinte();
        p.Afiseaza();

        Parinte p2=new Copil();
        p2.Afiseaza();

        Copil c=new Copil();
        c.Afiseaza();

        Parinte p3= new Parinte("Andrei");
        p3.AfiseazaNume();


        List<Parinte> lista=new ArrayList<Parinte>();
        lista.add(new Parinte("Alex"));
        lista.add(new Parinte("Robert"));
        lista.add(new Parinte("Sorin"));
        lista.add(new Parinte("Bert"));
        lista.add(new Parinte("Xyler"));
        Collections.sort(lista, new Comparator<Parinte>() {
            @Override
            public int compare(Parinte p1, Parinte p2) {
                return p1.Nume.compareTo(p2.Nume);
            }
        });

        for(Parinte Parinte:lista){
            System.out.println(Parinte.Nume);
        }
    }
}