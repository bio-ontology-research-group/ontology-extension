//--------------------------------------------------------
// this is a modified vesion of the code by @smalghamdi
//--------------------------------------------------------
@Grapes([
	  @Grab(group='org.slf4j', module='slf4j-simple', version='1.6.1'),
          @Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.3'),
          @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.5'),
          @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.5'),
          @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.2.5'),
          @Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.2.5')
        ])

import org.semanticweb.owlapi.model.parameters.*
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration
import org.semanticweb.elk.reasoner.config.*
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*
import org.semanticweb.owlapi.reasoner.structural.StructuralReasoner
import org.semanticweb.owlapi.vocab.OWLRDFVocabulary;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.owllink.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.owlapi.manchestersyntax.renderer.*;
import org.semanticweb.owlapi.reasoner.structural.*
import java.io.File;
import org.semanticweb.owlapi.util.OWLOntologyMerger;


OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLDataFactory fac = manager.getOWLDataFactory()

ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)
ElkReasonerFactory f = new ElkReasonerFactory()


def doid = manager.loadOntologyFromOntologyDocument(new File('/scratch/dragon/intel/althubsw/Ontology/doid.owl'))
OWLReasoner maReasoner = f.createReasoner(doid,config)

OWLClass ind = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_0050117"))

OWLClass subind1 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_104"))
OWLClass subind2 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_1564"))
OWLClass subind3 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_1398"))
OWLClass subind4 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_934"))

OWLClass subind5 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_1287"))
OWLClass subind6 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_28"))
OWLClass subind7 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_77"))
OWLClass subind8 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_74"))

OWLClass subind9 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_2914"))
OWLClass subind10 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_16"))
OWLClass subind11 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_17"))
OWLClass subind12 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_863"))

OWLClass subind13 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_15"))
OWLClass subind14 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_1579"))
OWLClass subind15 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_0060118"))
OWLClass subind16 = fac.getOWLClass(IRI.create("http://purl.obolibrary.org/obo/DOID_18"))

OWLReasoner doReasoner = f.createReasoner(doid,config)
List<InferredAxiomGenerator<? extends OWLAxiom>> generator = new ArrayList<InferredAxiomGenerator<? extends OWLAxiom>>();
generator.add(new InferredSubClassAxiomGenerator());
generator.add(new InferredEquivalentClassAxiomGenerator());
InferredOntologyGenerator iog = new InferredOntologyGenerator(maReasoner, generator);
iog.fillOntology(fac, doid);

String fmt = "Test%02d.txt";
count = 0
def fsub1 = new File(String.format(fmt, 1));
def fsub2 = new File(String.format(fmt, 2));
def fsub3 = new File(String.format(fmt, 3));
def fsub4 = new File(String.format(fmt, 4));

def fsub5 = new File(String.format(fmt, 5));
def fsub6 = new File(String.format(fmt, 6));
def fsub7 = new File(String.format(fmt, 7));
def fsub8 = new File(String.format(fmt, 8));

def fsub9 = new File(String.format(fmt, 9));
def fsub10 = new File(String.format(fmt, 10));
def fsub11 = new File(String.format(fmt, 11));
def fsub12 = new File(String.format(fmt, 12));

def fsub13 = new File(String.format(fmt, 13));
def fsub14 = new File(String.format(fmt, 14));
def fsub15 = new File(String.format(fmt, 15));
def fsub16 = new File(String.format(fmt, 16));

doReasoner.getSubClasses(subind1, false).each{
        fsub1.append(it+"\n");      
}

doReasoner.getSubClasses(subind2, false).each{
        fsub2.append(it+"\n");       
}
doReasoner.getSubClasses(subind3, false).each{
        fsub3.append(it+"\n");  
}
doReasoner.getSubClasses(subind4, false).each{
        fsub4.append(it+"\n");
}
doReasoner.getSubClasses(subind5, false).each{
        fsub5.append(it+"\n");
}
doReasoner.getSubClasses(subind6, false).each{
        fsub6.append(it+"\n");
}
doReasoner.getSubClasses(subind7, false).each{
        fsub7.append(it+"\n");  
}
doReasoner.getSubClasses(subind8, false).each{
        fsub8.append(it+"\n");
}
doReasoner.getSubClasses(subind9, false).each{
        fsub9.append(it+"\n");
}
doReasoner.getSubClasses(subind10, false).each{
        fsub10.append(it+"\n");
}
doReasoner.getSubClasses(subind11, false).each{
        fsub11.append(it+"\n");
}
doReasoner.getSubClasses(subind12, false).each{
        fsub12.append(it+"\n");
}
doReasoner.getSubClasses(subind13, false).each{
        fsub13.append(it+"\n");
}
doReasoner.getSubClasses(subind14, false).each{
        fsub14.append(it+"\n");
}
doReasoner.getSubClasses(subind15, false).each{
        fsub15.append(it+"\n");
}
doReasoner.getSubClasses(subind16, false).each{
        fsub16.append(it+"\n");
}
