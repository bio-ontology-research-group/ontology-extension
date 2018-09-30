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


//load ontology
def doOnt = manager.loadOntologyFromOntologyDocument(new File('doid.owl'))
def file = new File('DISEASEMappingLB.txt')

doOnt.getClassesInSignature(true).each { cl ->
    EntitySearcher.getAnnotationAssertionAxioms(cl, doOnt).each { ax ->
    if(ax.toString().indexOf("label")>-1 )
{
	file << "---------------------------" + "\n"
	file << cl.toString() + "\n"
	file << ax.toString() + "\n"
    //println("---------------------------")
    //println(cl.toString())
    //println(ax.toString())
}
}

}
println("done")
