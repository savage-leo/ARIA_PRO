/*
 * ForexFeed.Net Data API
 *
 * Copyright 2009 ForexFeed.Net <copyright@forexfeed.net>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by ForexFeed.Net.
 * 4. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

import java.util.*;
import net.forexfeed.ForexFeed;

/**
 * @author ForexFeed.net
 */
public class ForexFeedExample {

  /* ------------------------------------------
   * EDIT THE FOLLOWING VARIABLES
   */
    
    
  /**
   * 
   * NOTE: You must replace "YOUR_APP_ID" below with your unique 15-character AppID
   *       which can  be received by logging into your account on our website at
   *       http://forexfeed.net
   */
    
  private static String app_id = "YOUR_APP_ID";
  private static String symbol = "EURUSD,GBPUSD,USDCHF,USDCAD,AUDUSD";
  private static int interval = 3600;
  private static int periods = 1;
  private static String price = "mid";

  /* END VARIABLES
   * ------------------------------------------
   */

  /* main method
   */
  public static void main(String[] args) {

    // Create the ForexFeed Object
    ForexFeed fxfeed = new ForexFeed(app_id, symbol, interval, periods, price);

    // Display a Conversion
    printConversion(fxfeed);

    // Display the Quotes
    printData(fxfeed);

    // Display the available Intervals
    printIntervals(fxfeed);

    // Display the available Symbols
    printSymbols(fxfeed);
  }

    /*
     *  Get a conversion and print it to System.out
     */
    private static void printConversion(ForexFeed fxfeed) {

        HashMap conversion = fxfeed.getConversion("EUR", "USD", 1);
        // HashMap conversion = fxfeed.getConversion("USD", "EUR", 1);

        System.out.println("-------- Conversion --------");
        if (fxfeed.getStatus().equals("OK")) {
            System.out.print(conversion.get("convert_value") + " ");
            System.out.print(conversion.get("convert_from") + " = ");
            System.out.print(conversion.get("conversion_value") + " ");
            System.out.print(conversion.get("convert_to") + " ");
            System.out.println("(rate: " + conversion.get("conversion_rate") + ")");
            System.out.println("");
        }
        else {
            System.out.println(("Status: " + fxfeed.getStatus()));
            System.out.println(("ErrorCode: " + fxfeed.getErrorCode()));
            System.out.println(("ErrorMessage: " + fxfeed.getErrorMessage()));
        }
    }

  /**
   * Get the data and print it to System.out
   */
  private static void printData(ForexFeed fxfeed) {

    /*
     * Fetch the Data
     */
    ArrayList quotes = fxfeed.getData();

    System.out.println("-------- Quotes --------");
    if (fxfeed.getStatus().equals("OK")) {

      System.out.println("Number of Quotes: " + fxfeed.getNumQuotes());
      System.out.println("Copyright: " + fxfeed.getCopyright());
      System.out.println("Website: " + fxfeed.getWebsite());
      System.out.println("License: " + fxfeed.getLicense());
      System.out.println("Redistribution: " + fxfeed.getRedistribution());
      System.out.println("AccessPeriod: " + fxfeed.getAccessPeriod());
      System.out.println("AccessPerPeriod: " + fxfeed.getAccessPerPeriod());
      System.out.println("AccessThisPeriod: " + fxfeed.getAccessThisPeriod());
      System.out.println("AccessRemainingThisPeriod: " + fxfeed.getAccessPeriodRemaining());
      System.out.println("AccessPeriodBegan: " + fxfeed.getAccessPeriodBegan());
      System.out.println("NextAccessPeriodStarts: " + fxfeed.getAccessPeriodStarts());

      /*
       * Get an Iterator object for the quotes ArrayList using iterator() method.
       */
      Iterator itr = quotes.iterator();

      /*
       * Iterate through the ArrayList iterator
       */
      System.out.println("----------------------------------------");
      System.out.println("Iterating through Quotes...");
      System.out.println("----------------------------------------");
      while (itr.hasNext()) {
        HashMap quote = (HashMap) itr.next();

        System.out.println("Quote Symbol: " + quote.get("symbol"));
        System.out.println("Title: " + quote.get("title"));
        System.out.println("Time: " + quote.get("time"));

        if (fxfeed.getInterval() == 1) {
          if (fxfeed.getPrice().equals("bid,ask")) {
            System.out.println("Bid: " + quote.get("bid"));
            System.out.println("Ask: " + quote.get("ask"));
          }
          else {
            System.out.println(" Price: " + quote.get("price"));
          }
        }
        else {
          System.out.println("Open: " + quote.get("open"));
          System.out.println("High: " + quote.get("high"));
          System.out.println("Low: " + quote.get("low"));
          System.out.println("Close: " + quote.get("close"));
        }
        System.out.println("");

      }

    }
    else {
      System.out.println("Status: " + fxfeed.getStatus());
      System.out.println("ErrorCode: " + fxfeed.getErrorCode());
      System.out.println("ErrorMessage: " + fxfeed.getErrorMessage());
    }

  }

  /**
   *  Print the Intervals to System.out
   */
  private static void printIntervals(ForexFeed fxfeed) {

    /*
     * Fetch the Intervals
     */
    HashMap intervals = fxfeed.getAvailableIntervals(false);


    System.out.println("-------- Intervals --------");
    if (fxfeed.getStatus().equals("OK")) {

      /*
       * Get a Collection of values contained in HashMap
       */
      Collection c = intervals.values();

      /*
       * Obtain an Iterator for Collection
       */
      Iterator itr = c.iterator();

      /*
       * Iterate through the HashMap values iterator
       */
      while (itr.hasNext()) {
        HashMap value = (HashMap) itr.next();
        System.out.println("Interval: " + value.get("interval"));
        System.out.println("Title: " + value.get("title"));
        System.out.println("");
      }
    }
    else {
      System.out.println("Status: " + fxfeed.getStatus());
      System.out.println("ErrorCode: " + fxfeed.getErrorCode());
      System.out.println("ErrorMessage: " + fxfeed.getErrorMessage());
    }

  }

  /**
   *  Print the Symbols to System.out
   */
  private static void printSymbols(ForexFeed fxfeed) {

    /*
     * Fetch the Symbols
     */
    HashMap symbols = fxfeed.getAvailableSymbols(false);

    System.out.println("-------- Symbols --------");
    if (fxfeed.getStatus().equals("OK")) {

      /*
       * Get a Collection of values contained in HashMap
       */
      Collection c = symbols.values();

      /*
       * Obtain an Iterator for Collection
       */
      Iterator itr = c.iterator();

      /*
       * Iterate through the HashMap values iterator
       */
      while (itr.hasNext()) {
        HashMap value = (HashMap) itr.next();
        System.out.println("Symbol: " + value.get("symbol"));
        System.out.println("Title: " + value.get("title"));
        System.out.println("Decimals: " + value.get("decimals"));
        System.out.println("");
      }
    }
    else {
      System.out.println("Status: " + fxfeed.getStatus());
      System.out.println("ErrorCode: " + fxfeed.getErrorCode());
      System.out.println("ErrorMessage: " + fxfeed.getErrorMessage());
    }

  }
} // end class

